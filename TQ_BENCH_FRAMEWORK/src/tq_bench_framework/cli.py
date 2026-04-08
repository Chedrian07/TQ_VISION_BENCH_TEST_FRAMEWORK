from __future__ import annotations

import argparse
import logging
from typing import Sequence

from tq_bench_framework.runner import RunOptions, execute_run, print_benchmark_list
from tq_bench_framework.settings import load_framework_settings


def _parse_bits_list(raw: str) -> tuple[float, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return tuple(values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tq-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-benchmarks", help="List benchmark manifests")
    list_parser.set_defaults(command="list-benchmarks")

    check_parser = subparsers.add_parser("check-datasets", help="Check dataset local/remote availability")
    check_parser.add_argument("--benchmarks", default="all")
    check_parser.add_argument("--benchmark", action="append", default=[])
    check_parser.set_defaults(command="check-datasets")

    prepare_parser = subparsers.add_parser("prepare-datasets", help="Download and normalize datasets into local JSONL files")
    prepare_parser.add_argument("--benchmarks", default="all")
    prepare_parser.add_argument("--benchmark", action="append", default=[])
    prepare_parser.add_argument("--num", type=int, default=None, help="Optional cap on the number of samples to materialize per benchmark")
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(command="prepare-datasets")

    precompute_parser = subparsers.add_parser(
        "precompute-vision",
        help="Warm the backend's vision feature cache by sending one tiny request per sample",
    )
    precompute_parser.add_argument("--benchmarks", default="all")
    precompute_parser.add_argument("--benchmark", action="append", default=[])
    precompute_parser.add_argument("--num", type=int, default=None, help="Per-benchmark sample cap")
    precompute_parser.add_argument("--seed", type=int, default=7)
    precompute_parser.add_argument("--dataset-file", action="append", default=[])
    precompute_parser.add_argument("--log-level", default="INFO")
    precompute_parser.set_defaults(command="precompute-vision")

    run_parser = subparsers.add_parser("run", help="Run selected benchmarks")
    run_parser.add_argument("--benchmarks", default="all", help="Comma-separated benchmark ids or 'all'")
    run_parser.add_argument("--benchmark", action="append", default=[], help="Repeatable benchmark selector")
    run_parser.add_argument("--num", "--nums", dest="num_limit", type=int, default=None, help="Per-benchmark sample cap")
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument("--run-name", default=None)
    run_parser.add_argument("--sampling-profile", choices=("controlled", "benchmark", "best_effort_general", "best_effort_reasoning"), default="controlled")
    run_parser.add_argument("--model", default=None, help="Override backend model for this run")
    run_parser.add_argument("--revision", default=None)
    run_parser.add_argument("--adapter-path", default=None)
    run_parser.add_argument("--dataset-file", action="append", default=[], help="benchmark_id=/abs/path/file.jsonl")
    run_parser.add_argument("--max-output-tokens", type=int, default=None)
    run_parser.add_argument("--mlx-bits", default="2,3,4")
    run_parser.add_argument("--turboquant-bits", default="2,2.5,3,3.5,4")
    run_parser.add_argument("--no-baseline", action="store_true")
    run_parser.add_argument("--no-mlx", action="store_true")
    run_parser.add_argument("--no-turboquant", action="store_true")
    run_parser.add_argument("--no-resume", action="store_true")
    run_parser.add_argument("--resume-run-id", default=None, help="Resume into an existing run id under results/runs/<run_id>")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument(
        "--in-process",
        action="store_true",
        help=(
            "Bypass the HTTP backend and run oMLX directly in-process. "
            "Faster (no HTTP/SSE serialization) but cannot share state "
            "with an external server. The TQ_BACKEND server is not used."
        ),
    )
    run_parser.add_argument("--log-level", default="INFO")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(getattr(args, "log_level", "INFO")).upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "list-benchmarks":
        print_benchmark_list()
        return 0
    if args.command == "check-datasets":
        from tq_bench_framework.dataset_prepare import check_dataset

        settings = load_framework_settings()
        selected = _resolve_prepare_selection(args.benchmarks, args.benchmark)
        for benchmark_id in selected:
            result = check_dataset(settings, benchmark_id)
            print(
                f"{result.benchmark_id:14} local={'yes' if result.local_ready else 'no '} "
                f"remote={'yes' if result.remote_ok else 'no '} "
                f"file={result.local_path or 'n/a'} "
                f"source={result.repo_id}:{result.split}"
            )
            if result.note:
                print(f"  note: {result.note}")
            print(f"  remote_message: {result.remote_message}")
        return 0
    if args.command == "prepare-datasets":
        from tq_bench_framework.dataset_prepare import prepare_dataset

        settings = load_framework_settings()
        selected = _resolve_prepare_selection(args.benchmarks, args.benchmark)
        for benchmark_id in selected:
            path = prepare_dataset(
                settings,
                benchmark_id,
                num_limit=args.num,
                overwrite=args.overwrite,
            )
            print(f"{benchmark_id}: prepared -> {path}")
        return 0
    if args.command == "precompute-vision":
        from tq_bench_framework.runner import precompute_vision_cache

        return precompute_vision_cache(
            benchmark_ids=[item for item in [args.benchmarks, *args.benchmark] if item],
            num_limit=args.num,
            seed=args.seed,
            dataset_file_overrides=list(args.dataset_file),
        )

    options = RunOptions(
        benchmark_ids=[item for item in [args.benchmarks, *args.benchmark] if item],
        num_limit=args.num_limit,
        seed=args.seed,
        run_name=args.run_name,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
        sampling_profile_mode=args.sampling_profile,
        model=args.model,
        revision=args.revision,
        adapter_path=args.adapter_path,
        dataset_file_overrides=list(args.dataset_file),
        include_baseline=not args.no_baseline,
        include_mlx=not args.no_mlx,
        include_turboquant=not args.no_turboquant,
        mlx_bits=_parse_bits_list(args.mlx_bits),
        turboquant_bits=_parse_bits_list(args.turboquant_bits),
        max_output_tokens_override=args.max_output_tokens,
        dry_run=args.dry_run,
        resume_run_id=args.resume_run_id,
        in_process=args.in_process,
    )
    return execute_run(options)


def _resolve_prepare_selection(benchmark_csv: str | None, repeated: list[str]) -> list[str]:
    from tq_bench_framework.dataset_prepare import available_source_ids

    selected: list[str] = []
    if benchmark_csv:
        selected.extend(item.strip() for item in benchmark_csv.split(",") if item.strip())
    selected.extend(item.strip() for item in repeated if item.strip())
    if not selected or selected == ["all"]:
        return available_source_ids()
    if "all" in selected and len(selected) > 1:
        selected = [item for item in selected if item != "all"]
    return list(dict.fromkeys(selected))
