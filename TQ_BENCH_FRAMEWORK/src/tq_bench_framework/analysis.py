from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SummaryRecord:
    benchmark_id: str
    benchmark_title: str
    runtime_label: str
    quant_scheme: str
    quant_bits: float | None
    sampling_profile: str
    metric: str
    num_samples: int
    num_scored: int
    num_errors: int
    mean_score: float
    mean_ttft_ms: float | None
    mean_total_latency_ms: float
    mean_decode_tps: float | None
    mean_prompt_tokens: float
    mean_output_tokens: float


def _load_summary_records(summary_csv: Path) -> list[SummaryRecord]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    records: list[SummaryRecord] = []
    for row in rows:
        records.append(
            SummaryRecord(
                benchmark_id=row["benchmark_id"],
                benchmark_title=row["benchmark_title"],
                runtime_label=row["runtime_label"],
                quant_scheme=row["quant_scheme"],
                quant_bits=float(row["quant_bits"]) if row["quant_bits"] not in ("", "None") else None,
                sampling_profile=row["sampling_profile"],
                metric=row["metric"],
                num_samples=int(row["num_samples"]),
                num_scored=int(row["num_scored"]),
                num_errors=int(row["num_errors"]),
                mean_score=float(row["mean_score"]),
                mean_ttft_ms=float(row["mean_ttft_ms"]) if row["mean_ttft_ms"] not in ("", "None") else None,
                mean_total_latency_ms=float(row["mean_total_latency_ms"]),
                mean_decode_tps=float(row["mean_decode_tps"]) if row["mean_decode_tps"] not in ("", "None") else None,
                mean_prompt_tokens=float(row["mean_prompt_tokens"]),
                mean_output_tokens=float(row["mean_output_tokens"]),
            )
        )
    return records


def _group_by_benchmark(records: list[SummaryRecord]) -> dict[str, list[SummaryRecord]]:
    grouped: dict[str, list[SummaryRecord]] = {}
    for record in records:
        grouped.setdefault(record.benchmark_id, []).append(record)
    return grouped


def _analysis_payload(records: list[SummaryRecord]) -> dict[str, Any]:
    grouped = _group_by_benchmark(records)
    by_benchmark: dict[str, Any] = {}
    for benchmark_id, items in grouped.items():
        baseline = next((item for item in items if item.quant_scheme == "none"), None)
        sorted_items = sorted(items, key=lambda item: item.mean_score, reverse=True)
        top = sorted_items[0] if sorted_items else None
        benchmark_payload = {
            "benchmark_id": benchmark_id,
            "benchmark_title": items[0].benchmark_title,
            "metric": items[0].metric,
            "baseline": asdict(baseline) if baseline else None,
            "best_runtime": asdict(top) if top else None,
            "runtimes": [],
        }
        baseline_score = baseline.mean_score if baseline else None
        for item in sorted(items, key=lambda row: row.runtime_label):
            row_payload = asdict(item)
            row_payload["score_delta_vs_baseline"] = (
                item.mean_score - baseline_score if baseline_score is not None else None
            )
            benchmark_payload["runtimes"].append(row_payload)
        by_benchmark[benchmark_id] = benchmark_payload

    return {
        "num_records": len(records),
        "benchmarks": by_benchmark,
    }


def _write_analysis_md(records: list[SummaryRecord], report_dir: Path) -> None:
    payload = _analysis_payload(records)
    lines = [
        "# Analysis Notes",
        "",
        "## Per-benchmark Best Runtime",
        "",
        "| Benchmark | Best Runtime | Score | Baseline | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for benchmark_id, section in sorted(payload["benchmarks"].items()):
        best = section["best_runtime"]
        baseline = section["baseline"]
        best_score = best["mean_score"] if best else 0.0
        baseline_score = baseline["mean_score"] if baseline else 0.0
        delta = best_score - baseline_score
        lines.append(
            f"| {benchmark_id} | {best['runtime_label'] if best else 'n/a'} | "
            f"{best_score:.4f} | {baseline_score:.4f} | {delta:+.4f} |"
        )

    (report_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_score_heatmap(records: list[SummaryRecord], report_dir: Path) -> None:
    benchmarks = sorted({record.benchmark_id for record in records})
    runtimes = sorted({record.runtime_label for record in records})
    grid = [[math.nan for _ in runtimes] for _ in benchmarks]

    benchmark_index = {name: idx for idx, name in enumerate(benchmarks)}
    runtime_index = {name: idx for idx, name in enumerate(runtimes)}
    for record in records:
        grid[benchmark_index[record.benchmark_id]][runtime_index[record.runtime_label]] = record.mean_score

    fig, ax = plt.subplots(figsize=(max(8, len(runtimes) * 0.8), max(4, len(benchmarks) * 0.6)))
    image = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(runtimes)))
    ax.set_xticklabels(runtimes, rotation=45, ha="right")
    ax.set_yticks(range(len(benchmarks)))
    ax.set_yticklabels(benchmarks)
    ax.set_title("Mean Score Heatmap")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean score")
    fig.tight_layout()
    fig.savefig(report_dir / "score_heatmap.png", dpi=160)
    plt.close(fig)


def _plot_ttft_heatmap(records: list[SummaryRecord], report_dir: Path) -> None:
    ttft_records = [record for record in records if record.mean_ttft_ms is not None]
    if not ttft_records:
        return

    benchmarks = sorted({record.benchmark_id for record in ttft_records})
    runtimes = sorted({record.runtime_label for record in ttft_records})
    grid = [[math.nan for _ in runtimes] for _ in benchmarks]
    benchmark_index = {name: idx for idx, name in enumerate(benchmarks)}
    runtime_index = {name: idx for idx, name in enumerate(runtimes)}
    for record in ttft_records:
        grid[benchmark_index[record.benchmark_id]][runtime_index[record.runtime_label]] = record.mean_ttft_ms or math.nan

    fig, ax = plt.subplots(figsize=(max(8, len(runtimes) * 0.8), max(4, len(benchmarks) * 0.6)))
    image = ax.imshow(grid, aspect="auto", cmap="magma_r")
    ax.set_xticks(range(len(runtimes)))
    ax.set_xticklabels(runtimes, rotation=45, ha="right")
    ax.set_yticks(range(len(benchmarks)))
    ax.set_yticklabels(benchmarks)
    ax.set_title("Mean TTFT Heatmap (ms)")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("TTFT (ms)")
    fig.tight_layout()
    fig.savefig(report_dir / "ttft_heatmap.png", dpi=160)
    plt.close(fig)


def _plot_latency_vs_score(records: list[SummaryRecord], report_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "none": "#4C566A",
        "mlx": "#1F77B4",
        "turboquant": "#D62728",
    }
    for record in records:
        ax.scatter(
            record.mean_total_latency_ms,
            record.mean_score,
            color=colors.get(record.quant_scheme, "#2CA02C"),
            label=record.quant_scheme,
            alpha=0.8,
        )
        ax.annotate(
            f"{record.benchmark_id}:{record.runtime_label}",
            (record.mean_total_latency_ms, record.mean_score),
            fontsize=7,
            alpha=0.8,
        )
    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    ax.legend(dedup.values(), dedup.keys(), loc="best")
    ax.set_xlabel("Mean total latency (ms)")
    ax.set_ylabel("Mean score")
    ax.set_title("Latency vs Score")
    fig.tight_layout()
    fig.savefig(report_dir / "latency_vs_score.png", dpi=160)
    plt.close(fig)


def _plot_runtime_scores(records: list[SummaryRecord], report_dir: Path) -> None:
    benchmarks = sorted({record.benchmark_id for record in records})
    runtime_labels = sorted({record.runtime_label for record in records})

    fig, ax = plt.subplots(figsize=(max(10, len(benchmarks) * 1.5), 6))
    x_positions = list(range(len(benchmarks)))
    width = 0.8 / max(len(runtime_labels), 1)

    for index, runtime_label in enumerate(runtime_labels):
        values = []
        for benchmark in benchmarks:
            match = next(
                (record.mean_score for record in records if record.benchmark_id == benchmark and record.runtime_label == runtime_label),
                math.nan,
            )
            values.append(match)
        offsets = [x + (index - (len(runtime_labels) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=runtime_label)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(benchmarks, rotation=35, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_title("Benchmark Score by Runtime")
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(report_dir / "runtime_scores.png", dpi=160)
    plt.close(fig)


def generate_analysis_artifacts(summary_csv: Path, report_dir: Path) -> None:
    if not summary_csv.exists():
        return
    records = _load_summary_records(summary_csv)
    if not records:
        return

    report_dir.mkdir(parents=True, exist_ok=True)
    payload = _analysis_payload(records)
    (report_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_analysis_md(records, report_dir)
    _plot_score_heatmap(records, report_dir)
    _plot_ttft_heatmap(records, report_dir)
    _plot_latency_vs_score(records, report_dir)
    _plot_runtime_scores(records, report_dir)
