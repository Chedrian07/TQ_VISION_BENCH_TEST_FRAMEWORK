from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

from tq_bench_framework.benchmarks.registry import load_benchmark_registry
from tq_bench_framework.dataset import (
    DatasetError,
    count_samples,
    iter_selected_samples,
    parse_dataset_file_overrides,
    resolve_dataset_file,
)
from tq_bench_framework.metrics import score_prediction
from tq_bench_framework.reporting import (
    RunLogger,
    SummaryAccumulator,
    finalize_cell_summary,
)
from tq_bench_framework.runtime_client import BackendClient, make_backend_client
from tq_bench_framework.schema import (
    BenchmarkManifest,
    RunMetadata,
    RuntimeConfig,
    SampleResult,
)
from tq_bench_framework.settings import FrameworkSettings, load_framework_settings

log = logging.getLogger("tq-bench")


@dataclass(frozen=True)
class RunOptions:
    benchmark_ids: list[str]
    num_limit: int | None
    seed: int
    run_name: str | None
    resume: bool
    fail_fast: bool
    sampling_profile_mode: str
    model: str | None
    revision: str | None
    adapter_path: str | None
    dataset_file_overrides: list[str]
    include_baseline: bool
    include_mlx: bool
    include_turboquant: bool
    mlx_bits: tuple[float, ...]
    turboquant_bits: tuple[float, ...]
    max_output_tokens_override: int | None
    dry_run: bool
    resume_run_id: str | None
    in_process: bool = False


def parse_benchmark_selection(
    registry: dict[str, BenchmarkManifest],
    benchmark_csv: str | None,
    repeated_benchmarks: list[str],
) -> list[str]:
    selected: list[str] = []
    if benchmark_csv:
        selected.extend(item.strip() for item in benchmark_csv.split(",") if item.strip())
    selected.extend(item.strip() for item in repeated_benchmarks if item.strip())
    if not selected or selected == ["all"]:
        return list(registry.keys())
    if "all" in selected and len(selected) > 1:
        selected = [item for item in selected if item != "all"]

    unknown = [item for item in selected if item not in registry]
    if unknown:
        raise ValueError(f"Unknown benchmark ids: {', '.join(sorted(unknown))}")
    return list(dict.fromkeys(selected))


def resolve_sampling_profile(mode: str, manifest: BenchmarkManifest) -> str:
    if mode == "benchmark":
        return manifest.best_effort_sampling_profile
    return mode


def build_runtime_matrix(options: RunOptions, manifests: Iterable[BenchmarkManifest]) -> list[RuntimeConfig]:
    profiles = {
        resolve_sampling_profile(options.sampling_profile_mode, manifest)
        for manifest in manifests
    }

    matrix: list[RuntimeConfig] = []
    for profile in sorted(profiles):
        if options.include_baseline:
            matrix.append(
                RuntimeConfig(
                    scheme="none",
                    bits=None,
                    sampling_profile=profile,
                    model=options.model,
                    revision=options.revision,
                    adapter_path=options.adapter_path,
                )
            )
        if options.include_mlx:
            for bits in options.mlx_bits:
                matrix.append(
                    RuntimeConfig(
                        scheme="mlx",
                        bits=bits,
                        sampling_profile=profile,
                        model=options.model,
                        revision=options.revision,
                        adapter_path=options.adapter_path,
                    )
                )
        if options.include_turboquant:
            for bits in options.turboquant_bits:
                matrix.append(
                    RuntimeConfig(
                        scheme="turboquant",
                        bits=bits,
                        sampling_profile=profile,
                        model=options.model,
                        revision=options.revision,
                        adapter_path=options.adapter_path,
                    )
                )
    return matrix


def build_prompt(manifest: BenchmarkManifest, question: str, metadata: dict) -> str:
    try:
        return manifest.prompt_template.format(question=question, **metadata)
    except (KeyError, IndexError, ValueError) as exc:
        log.warning(
            "Prompt template formatting failed for benchmark=%s with metadata keys=%s: %s",
            manifest.id,
            sorted(metadata.keys()),
            exc,
        )
        return question


def _verify_runtime_state(runtime: RuntimeConfig, runtime_state: dict) -> None:
    if not runtime_state.get("loaded"):
        raise RuntimeError(f"Backend runtime '{runtime.label}' is not loaded after reload.")

    kv_cache = runtime_state.get("kv_cache") or {}
    actual_scheme = kv_cache.get("scheme")
    if actual_scheme != runtime.scheme:
        raise RuntimeError(
            f"Backend runtime reload mismatch for {runtime.label}: scheme={actual_scheme!r} expected={runtime.scheme!r}."
        )

    actual_bits = kv_cache.get("bits")
    if runtime.bits is None:
        if actual_bits not in (None, "", "None"):
            raise RuntimeError(
                f"Backend runtime reload mismatch for {runtime.label}: bits={actual_bits!r} expected=None."
            )
    elif actual_bits is None or float(actual_bits) != float(runtime.bits):
        raise RuntimeError(
            f"Backend runtime reload mismatch for {runtime.label}: bits={actual_bits!r} expected={runtime.bits!r}."
        )

    actual_profile = runtime_state.get("sampling_profile")
    if actual_profile != runtime.sampling_profile:
        raise RuntimeError(
            f"Backend runtime reload mismatch for {runtime.label}: sampling_profile={actual_profile!r} "
            f"expected={runtime.sampling_profile!r}."
        )

    model_state = runtime_state.get("model") or {}
    if runtime.model is not None and model_state.get("id") != runtime.model:
        raise RuntimeError(
            f"Backend runtime reload mismatch for {runtime.label}: model={model_state.get('id')!r} expected={runtime.model!r}."
        )
    if runtime.revision is not None and model_state.get("revision") != runtime.revision:
        raise RuntimeError(
            f"Backend runtime reload mismatch for {runtime.label}: revision={model_state.get('revision')!r} expected={runtime.revision!r}."
        )
    if runtime.adapter_path is not None and model_state.get("adapter_path") != runtime.adapter_path:
        raise RuntimeError(
            f"Backend runtime reload mismatch for {runtime.label}: adapter_path={model_state.get('adapter_path')!r} expected={runtime.adapter_path!r}."
        )


def _iter_pending_sample_batches(
    samples: Iterable,
    *,
    completed_ids: set[str],
    batch_size: int,
):
    batch: list = []
    for sample in samples:
        if sample.sample_id in completed_ids:
            continue
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _run_single_sample(
    *,
    client: BackendClient,
    run_id: str,
    manifest: BenchmarkManifest,
    runtime: RuntimeConfig,
    model_id: str,
    sample,
    max_output_tokens_override: int | None,
) -> tuple[SampleResult, Exception | None]:
    prompt = build_prompt(manifest, sample.question, sample.metadata)
    try:
        inference = client.stream_response(
            model=model_id,
            prompt=prompt,
            images=sample.images,
            max_output_tokens=max_output_tokens_override or manifest.max_output_tokens,
            system_prompt=manifest.system_prompt,
        )
        prediction = str(inference["output_text"]).strip()
        score = score_prediction(
            manifest.metric,
            prediction,
            sample.answers,
            sample.metadata,
        )
        return (
            SampleResult(
                run_id=run_id,
                benchmark_id=manifest.id,
                benchmark_title=manifest.title,
                sample_id=sample.sample_id,
                runtime_label=runtime.label,
                quant_scheme=runtime.scheme,
                quant_bits=runtime.bits,
                sampling_profile=runtime.sampling_profile,
                question=sample.question,
                answers=sample.answers,
                prediction=prediction,
                score=score,
                metric=manifest.metric,
                ttft_ms=inference["ttft_ms"],
                total_latency_ms=inference["total_latency_ms"],
                decode_latency_ms=inference["decode_latency_ms"],
                decode_tps=inference["decode_tps"],
                prompt_tokens=inference["prompt_tokens"],
                output_tokens=inference["output_tokens"],
                images=sample.images,
                metadata=sample.metadata,
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            SampleResult(
                run_id=run_id,
                benchmark_id=manifest.id,
                benchmark_title=manifest.title,
                sample_id=sample.sample_id,
                runtime_label=runtime.label,
                quant_scheme=runtime.scheme,
                quant_bits=runtime.bits,
                sampling_profile=runtime.sampling_profile,
                question=sample.question,
                answers=sample.answers,
                prediction="",
                score=0.0,
                metric=manifest.metric,
                ttft_ms=None,
                total_latency_ms=0.0,
                decode_latency_ms=None,
                decode_tps=None,
                prompt_tokens=0,
                output_tokens=0,
                images=sample.images,
                metadata=sample.metadata,
                error=str(exc),
            ),
            exc,
        )


def execute_run(options: RunOptions) -> int:
    settings = load_framework_settings()
    registry = load_benchmark_registry()
    selected_ids = parse_benchmark_selection(
        registry,
        benchmark_csv=",".join(options.benchmark_ids) if options.benchmark_ids else None,
        repeated_benchmarks=[],
    )
    manifests = [registry[benchmark_id] for benchmark_id in selected_ids]
    runtime_matrix = build_runtime_matrix(options, manifests)
    dataset_overrides = parse_dataset_file_overrides(options.dataset_file_overrides)
    if options.resume_run_id is not None:
        run_json = settings.results_dir / "runs" / options.resume_run_id / "run.json"
        if not run_json.exists():
            raise ValueError(
                f"Requested resume_run_id='{options.resume_run_id}' but no existing run metadata was found."
            )

    logger = RunLogger(
        results_root=settings.results_dir,
        reports_root=settings.reports_dir,
        run_name=options.run_name,
        resume_run_id=options.resume_run_id,
    )
    existing_metadata = logger.load_run_metadata()
    run_metadata = RunMetadata(
        run_id=logger.run_id,
        run_dir=logger.run_dir,
        selected_benchmarks=selected_ids,
        runtime_matrix=[config.label for config in runtime_matrix],
        num_limit=options.num_limit,
        seed=options.seed,
        sampling_profile_mode=options.sampling_profile_mode,
        resumed_from_run_id=options.resume_run_id,
        model=options.model,
        revision=options.revision,
        adapter_path=options.adapter_path,
    )
    if options.resume_run_id is not None and existing_metadata is not None:
        guards = {
            "num_limit": options.num_limit,
            "seed": options.seed,
            "sampling_profile_mode": options.sampling_profile_mode,
            "model": options.model,
            "revision": options.revision,
            "adapter_path": options.adapter_path,
        }
        mismatches: list[str] = []
        existing_selected = existing_metadata.get("selected_benchmarks") or []
        missing_benchmarks = [item for item in selected_ids if item not in existing_selected]
        if missing_benchmarks:
            mismatches.append(
                "selected_benchmarks: requested benchmarks are not a subset of the original run "
                f"(missing: {missing_benchmarks!r})"
            )
        existing_runtime_matrix = existing_metadata.get("runtime_matrix") or []
        missing_runtime_matrix = [
            item for item in [config.label for config in runtime_matrix]
            if item not in existing_runtime_matrix
        ]
        if missing_runtime_matrix:
            mismatches.append(
                "runtime_matrix: requested runtimes are not a subset of the original run "
                f"(missing: {missing_runtime_matrix!r})"
            )
        for key, value in guards.items():
            if existing_metadata.get(key) != value:
                mismatches.append(
                    f"{key}: existing={existing_metadata.get(key)!r} current={value!r}"
                )
        if mismatches:
            joined = "; ".join(mismatches)
            raise ValueError(
                "Resume parameters do not match the original run. "
                f"Refusing to resume: {joined}"
            )
    logger.write_run_metadata(run_metadata)

    if options.dry_run:
        logger.record_event(
            "dry_run",
            {
                "selected_benchmarks": selected_ids,
                "runtime_matrix": [config.reload_payload() for config in runtime_matrix],
            },
        )
        return 0

    client = make_backend_client(settings, in_process=options.in_process)
    if options.in_process:
        log.info("Embedded oMLX runtime active (HTTP backend bypassed)")
    executor = ThreadPoolExecutor(max_workers=settings.max_parallel_requests)
    try:
        runtime_groups: dict[tuple[str, str | None, float | None], list[tuple[BenchmarkManifest, RuntimeConfig]]] = defaultdict(list)
        for manifest in manifests:
            profile = resolve_sampling_profile(options.sampling_profile_mode, manifest)
            for runtime in runtime_matrix:
                if runtime.sampling_profile != profile:
                    continue
                runtime_groups[(runtime.scheme, runtime.sampling_profile, runtime.bits)].append((manifest, runtime))

        for _, grouped_jobs in runtime_groups.items():
            runtime = grouped_jobs[0][1]
            log.info("Reloading backend runtime: %s", runtime.label)
            logger.record_event("runtime_reload_start", {"runtime": runtime.reload_payload()})
            reload_payload = client.reload_runtime(runtime)
            logger.record_event("runtime_reload_done", reload_payload)

            runtime_state = reload_payload.get("current", {})
            _verify_runtime_state(runtime, runtime_state)
            model_id = str(runtime_state.get("model", {}).get("id") or runtime.model or "")
            for manifest, cell_runtime in grouped_jobs:
                dataset_file = resolve_dataset_file(manifest, dataset_overrides)
                dataset_row_count = count_samples(dataset_file)
                if dataset_row_count == 0:
                    raise DatasetError(
                        f"Dataset file '{dataset_file}' for benchmark '{manifest.id}' has no non-empty rows."
                    )
                samples = iter_selected_samples(
                    manifest,
                    dataset_file,
                    num_limit=options.num_limit,
                    seed=options.seed,
                )
                raw_path = logger.raw_results_path(manifest.id, cell_runtime.filename_label)
                completed_ids, accumulator = (
                    logger.restore_resume_state(raw_path)
                    if options.resume
                    else (set(), SummaryAccumulator())
                )
                total_selected = dataset_row_count if options.num_limit is None else min(dataset_row_count, options.num_limit)
                remaining_samples = max(total_selected - len(completed_ids), 0)
                progress_interval = max(50, remaining_samples // 20) if remaining_samples >= 50 else None
                logger.record_event(
                    "cell_start",
                    {
                        "benchmark_id": manifest.id,
                        "runtime_label": cell_runtime.label,
                        "num_selected_samples": options.num_limit,
                        "num_already_completed": len(completed_ids),
                        "dataset_file": str(dataset_file),
                    },
                )
                log.info(
                    "Running benchmark=%s runtime=%s dataset=%s num_limit=%s total_selected=%d remaining=%d parallel=%d",
                    manifest.id,
                    cell_runtime.label,
                    dataset_file,
                    options.num_limit,
                    total_selected,
                    remaining_samples,
                    settings.max_parallel_requests,
                )

                processed_in_cell = 0
                for sample_batch in _iter_pending_sample_batches(
                    samples,
                    completed_ids=completed_ids,
                    batch_size=settings.max_parallel_requests,
                ):
                    futures = [
                        executor.submit(
                            _run_single_sample,
                            client=client,
                            run_id=logger.run_id,
                            manifest=manifest,
                            runtime=cell_runtime,
                            model_id=model_id,
                            sample=sample,
                            max_output_tokens_override=options.max_output_tokens_override,
                        )
                        for sample in sample_batch
                    ]

                    batch_failure: Exception | None = None
                    for future in futures:
                        result, exc = future.result()
                        logger.append_sample_result(raw_path, result)
                        accumulator.update(result)
                        processed_in_cell += 1
                        if exc is not None and batch_failure is None:
                            batch_failure = exc

                        if progress_interval and (
                            processed_in_cell % progress_interval == 0
                            or processed_in_cell == remaining_samples
                        ):
                            log.info(
                                "Progress benchmark=%s runtime=%s %d/%d",
                                manifest.id,
                                cell_runtime.label,
                                processed_in_cell,
                                remaining_samples,
                            )

                    if batch_failure is not None and options.fail_fast:
                        raise batch_failure

                summary = finalize_cell_summary(
                    benchmark_id=manifest.id,
                    benchmark_title=manifest.title,
                    runtime_label=cell_runtime.label,
                    quant_scheme=cell_runtime.scheme,
                    quant_bits=cell_runtime.bits,
                    sampling_profile=cell_runtime.sampling_profile,
                    metric=manifest.metric,
                    accumulator=accumulator,
                )
                logger.append_cell_summary(summary)
                log.info(
                    "Completed benchmark=%s runtime=%s score=%.4f errors=%d samples=%d",
                    manifest.id,
                    cell_runtime.label,
                    summary.mean_score,
                    summary.num_errors,
                    summary.num_samples,
                )
                logger.record_event(
                    "cell_done",
                    {
                        "benchmark_id": manifest.id,
                        "runtime_label": cell_runtime.label,
                        "summary": summary.to_csv_row(),
                    },
                )
        logger.finalize_reports()
        return 0
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        client.close()


def print_benchmark_list() -> None:
    registry = load_benchmark_registry()
    for benchmark in registry.values():
        print(
            f"{benchmark.id:14} stage={benchmark.stage:13} metric={benchmark.metric:24} "
            f"default={benchmark.default_sampling_profile:10} best_effort={benchmark.best_effort_sampling_profile}"
        )


def precompute_vision_cache(
    *,
    benchmark_ids: list[str],
    num_limit: int | None,
    seed: int,
    dataset_file_overrides: list[str],
) -> int:
    """Warm the backend's ``VisionFeatureSSDCache`` for selected samples.

    Iterates the (benchmark, sample) plan once and sends a tiny
    ``max_output_tokens=1`` request per sample. The vision encoder runs
    once per unique image and the resulting features land in
    ``<TQ_OMLX_CACHE_DIR>/vision_features``. Subsequent ``tq-bench run``
    invocations (across runtime reloads, across processes) become cache
    hits and skip the vision tower entirely.

    The function intentionally does NOT touch the runtime config -
    whatever the backend currently has loaded is fine because the
    vision tower path is independent of the language-model KV quant
    scheme. Returns shell exit code (0 on success).
    """
    settings = load_framework_settings()
    registry = load_benchmark_registry()
    selected_ids = parse_benchmark_selection(
        registry,
        benchmark_csv=",".join(benchmark_ids) if benchmark_ids else None,
        repeated_benchmarks=[],
    )
    manifests = [registry[benchmark_id] for benchmark_id in selected_ids]
    dataset_overrides = parse_dataset_file_overrides(dataset_file_overrides)

    client = BackendClient(settings)
    executor = ThreadPoolExecutor(max_workers=settings.max_parallel_requests)
    total_samples = 0
    started = time.perf_counter()
    try:
        for manifest in manifests:
            dataset_file = resolve_dataset_file(manifest, dataset_overrides)
            dataset_row_count = count_samples(dataset_file)
            if dataset_row_count == 0:
                log.warning(
                    "precompute: benchmark=%s dataset=%s is empty, skipping",
                    manifest.id,
                    dataset_file,
                )
                continue
            samples = list(
                iter_selected_samples(
                    manifest,
                    dataset_file,
                    num_limit=num_limit,
                    seed=seed,
                )
            )
            total_for_bench = len(samples)
            log.info(
                "precompute: warming benchmark=%s samples=%d parallel=%d",
                manifest.id,
                total_for_bench,
                settings.max_parallel_requests,
            )

            def _warm_one(sample) -> tuple[str, Exception | None]:
                prompt = build_prompt(manifest, sample.question, sample.metadata)
                try:
                    client.stream_response(
                        model="",  # backend uses its loaded model regardless
                        prompt=prompt,
                        images=sample.images,
                        max_output_tokens=1,
                        system_prompt=manifest.system_prompt,
                    )
                    return sample.sample_id, None
                except Exception as exc:  # noqa: BLE001
                    return sample.sample_id, exc

            failures = 0
            for batch_start in range(0, total_for_bench, settings.max_parallel_requests):
                batch = samples[batch_start : batch_start + settings.max_parallel_requests]
                futures = [executor.submit(_warm_one, sample) for sample in batch]
                for future in futures:
                    sample_id, exc = future.result()
                    if exc is not None:
                        failures += 1
                        log.warning(
                            "precompute: benchmark=%s sample=%s failed: %s",
                            manifest.id,
                            sample_id,
                            exc,
                        )
            total_samples += total_for_bench
            log.info(
                "precompute: benchmark=%s done (%d/%d successful)",
                manifest.id,
                total_for_bench - failures,
                total_for_bench,
            )

        elapsed = time.perf_counter() - started
        log.info(
            "precompute complete: %d samples across %d benchmarks in %.1fs",
            total_samples,
            len(manifests),
            elapsed,
        )
        return 0
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        client.close()
