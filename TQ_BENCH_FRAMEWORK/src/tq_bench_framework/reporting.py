from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tq_bench_framework.analysis import generate_analysis_artifacts
from tq_bench_framework.schema import CellSummary, RunMetadata, SampleResult

log = logging.getLogger("tq-bench")


@dataclass
class SummaryAccumulator:
    count: int = 0
    scored_count: int = 0
    error_count: int = 0
    score_sum: float = 0.0
    ttft_sum: float = 0.0
    ttft_count: int = 0
    total_latency_sum: float = 0.0
    decode_tps_sum: float = 0.0
    decode_tps_count: int = 0
    prompt_tokens_sum: int = 0
    output_tokens_sum: int = 0

    def update(self, result: SampleResult) -> None:
        self.count += 1
        self.total_latency_sum += result.total_latency_ms
        self.prompt_tokens_sum += result.prompt_tokens
        self.output_tokens_sum += result.output_tokens
        if result.error:
            self.error_count += 1
            return
        self.scored_count += 1
        self.score_sum += result.score
        if result.ttft_ms is not None:
            self.ttft_sum += result.ttft_ms
            self.ttft_count += 1
        if result.decode_tps is not None:
            self.decode_tps_sum += result.decode_tps
            self.decode_tps_count += 1


class RunLogger:
    def __init__(
        self,
        *,
        results_root: Path,
        reports_root: Path,
        run_name: str | None,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{run_name}" if run_name else ""
        run_id = f"{timestamp}{suffix}"
        self.run_id = run_id
        self.run_dir = results_root / "runs" / run_id
        self.raw_dir = self.run_dir / "raw"
        self.logs_dir = self.run_dir / "logs"
        self.aggregate_dir = self.run_dir / "aggregate"
        self.report_dir = reports_root / run_id
        for directory in (
            self.run_dir,
            self.raw_dir,
            self.logs_dir,
            self.aggregate_dir,
            self.report_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.events_path = self.logs_dir / "events.jsonl"
        self.summary_csv_path = self.aggregate_dir / "summary.csv"
        self.summary_rows: list[CellSummary] = []

    def write_run_metadata(self, metadata: RunMetadata) -> None:
        path = self.run_dir / "run.json"
        path.write_text(json.dumps(metadata.to_json(), indent=2, ensure_ascii=False))

    def record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        row = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def raw_results_path(self, benchmark_id: str, runtime_filename_label: str) -> Path:
        return self.raw_dir / f"{benchmark_id}__{runtime_filename_label}.jsonl"

    def load_completed_sample_ids(self, path: Path) -> set[str]:
        if not path.exists():
            return set()
        completed: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample_id = row.get("sample_id")
                if sample_id:
                    completed.add(str(sample_id))
        return completed

    def append_sample_result(self, path: Path, result: SampleResult) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_json(), ensure_ascii=False) + "\n")

    def append_cell_summary(self, summary: CellSummary) -> None:
        self.summary_rows.append(summary)
        rows = [item.to_csv_row() for item in self.summary_rows]
        if not rows:
            return

        fieldnames = list(rows[0].keys())
        with self.summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        self._write_markdown_summary()

    def _write_markdown_summary(self) -> None:
        lines = [
            "# Benchmark Summary",
            "",
            "| Benchmark | Runtime | Score | Errors | TTFT ms | Total ms | Decode tok/s |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in self.summary_rows:
            lines.append(
                "| {benchmark} | {runtime} | {score:.4f} | {errors} | {ttft} | {total} | {decode} |".format(
                    benchmark=row.benchmark_id,
                    runtime=row.runtime_label,
                    score=row.mean_score,
                    errors=row.num_errors,
                    ttft=f"{row.mean_ttft_ms:.2f}" if row.mean_ttft_ms is not None else "n/a",
                    total=f"{row.mean_total_latency_ms:.2f}",
                    decode=f"{row.mean_decode_tps:.2f}" if row.mean_decode_tps is not None else "n/a",
                )
            )
        (self.report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    def finalize_reports(self) -> None:
        generate_analysis_artifacts(self.summary_csv_path, self.report_dir)


def finalize_cell_summary(
    *,
    benchmark_id: str,
    benchmark_title: str,
    runtime_label: str,
    quant_scheme: str,
    quant_bits: float | None,
    sampling_profile: str,
    metric: str,
    accumulator: SummaryAccumulator,
) -> CellSummary:
    return CellSummary(
        benchmark_id=benchmark_id,
        benchmark_title=benchmark_title,
        runtime_label=runtime_label,
        quant_scheme=quant_scheme,
        quant_bits=quant_bits,
        sampling_profile=sampling_profile,
        metric=metric,
        num_samples=accumulator.count,
        num_scored=accumulator.scored_count,
        num_errors=accumulator.error_count,
        mean_score=(accumulator.score_sum / accumulator.scored_count)
        if accumulator.scored_count
        else 0.0,
        mean_ttft_ms=(accumulator.ttft_sum / accumulator.ttft_count)
        if accumulator.ttft_count
        else None,
        mean_total_latency_ms=(accumulator.total_latency_sum / accumulator.count)
        if accumulator.count
        else 0.0,
        mean_decode_tps=(accumulator.decode_tps_sum / accumulator.decode_tps_count)
        if accumulator.decode_tps_count
        else None,
        mean_prompt_tokens=(accumulator.prompt_tokens_sum / accumulator.count)
        if accumulator.count
        else 0.0,
        mean_output_tokens=(accumulator.output_tokens_sum / accumulator.count)
        if accumulator.count
        else 0.0,
    )
