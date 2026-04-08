from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from tq_bench_framework.benchmarks.registry import load_benchmark_registry
from tq_bench_framework.dataset import (
    DatasetError,
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
from tq_bench_framework.runtime_client import BackendClient
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
            "selected_benchmarks": selected_ids,
            "runtime_matrix": [config.label for config in runtime_matrix],
            "num_limit": options.num_limit,
            "seed": options.seed,
            "sampling_profile_mode": options.sampling_profile_mode,
            "model": options.model,
            "revision": options.revision,
            "adapter_path": options.adapter_path,
        }
        mismatches: list[str] = []
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

    client = BackendClient(settings)
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
            model_id = str(runtime_state.get("model", {}).get("id") or runtime.model or "")
            for manifest, cell_runtime in grouped_jobs:
                dataset_file = resolve_dataset_file(manifest, dataset_overrides)
                samples = iter_selected_samples(
                    manifest,
                    dataset_file,
                    num_limit=options.num_limit,
                    seed=options.seed,
                )
                raw_path = logger.raw_results_path(manifest.id, cell_runtime.filename_label)
                completed_ids = logger.load_completed_sample_ids(raw_path) if options.resume else set()
                accumulator = logger.restore_accumulator(raw_path) if options.resume else SummaryAccumulator()
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
                    "Running benchmark=%s runtime=%s dataset=%s num_limit=%s",
                    manifest.id,
                    cell_runtime.label,
                    dataset_file,
                    options.num_limit,
                )

                for sample in samples:
                    if sample.sample_id in completed_ids:
                        continue

                    prompt = build_prompt(manifest, sample.question, sample.metadata)
                    try:
                        inference = client.stream_response(
                            model=model_id,
                            prompt=prompt,
                            images=sample.images,
                            max_output_tokens=options.max_output_tokens_override
                            or manifest.max_output_tokens,
                            system_prompt=manifest.system_prompt,
                        )
                        prediction = str(inference["output_text"]).strip()
                        score = score_prediction(
                            manifest.metric,
                            prediction,
                            sample.answers,
                            sample.metadata,
                        )
                        result = SampleResult(
                            run_id=logger.run_id,
                            benchmark_id=manifest.id,
                            benchmark_title=manifest.title,
                            sample_id=sample.sample_id,
                            runtime_label=cell_runtime.label,
                            quant_scheme=cell_runtime.scheme,
                            quant_bits=cell_runtime.bits,
                            sampling_profile=cell_runtime.sampling_profile,
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
                        )
                    except Exception as exc:  # noqa: BLE001
                        result = SampleResult(
                            run_id=logger.run_id,
                            benchmark_id=manifest.id,
                            benchmark_title=manifest.title,
                            sample_id=sample.sample_id,
                            runtime_label=cell_runtime.label,
                            quant_scheme=cell_runtime.scheme,
                            quant_bits=cell_runtime.bits,
                            sampling_profile=cell_runtime.sampling_profile,
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
                        )
                        if options.fail_fast:
                            logger.append_sample_result(raw_path, result)
                            accumulator.update(result)
                            raise

                    logger.append_sample_result(raw_path, result)
                    accumulator.update(result)

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
        client.close()


def print_benchmark_list() -> None:
    registry = load_benchmark_registry()
    for benchmark in registry.values():
        print(
            f"{benchmark.id:14} stage={benchmark.stage:13} metric={benchmark.metric:24} "
            f"default={benchmark.default_sampling_profile:10} best_effort={benchmark.best_effort_sampling_profile}"
        )
