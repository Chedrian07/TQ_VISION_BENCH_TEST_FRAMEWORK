from __future__ import annotations

import csv
import json
import logging
from dataclasses import MISSING, asdict, dataclass, fields
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
    total_latency_count: int = 0
    decode_tps_sum: float = 0.0
    decode_tps_count: int = 0
    prompt_tokens_sum: int = 0
    prompt_tokens_count: int = 0
    output_tokens_sum: int = 0
    output_tokens_count: int = 0

    def update(self, result: SampleResult) -> None:
        self.count += 1
        if result.error:
            self.error_count += 1
            return
        self.scored_count += 1
        self.score_sum += result.score
        self.total_latency_sum += result.total_latency_ms
        self.total_latency_count += 1
        self.prompt_tokens_sum += result.prompt_tokens
        self.prompt_tokens_count += 1
        self.output_tokens_sum += result.output_tokens
        self.output_tokens_count += 1
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
        resume_run_id: str | None = None,
    ):
        if resume_run_id is not None:
            run_id = resume_run_id
        else:
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
        self.metadata_updates_path = self.logs_dir / "run_metadata_updates.jsonl"
        self.summary_csv_path = self.aggregate_dir / "summary.csv"
        self.summary_rows: list[CellSummary] = self._load_existing_summaries()

    def _load_existing_summaries(self) -> list[CellSummary]:
        if not self.summary_csv_path.exists():
            return []
        rows: list[CellSummary] = []
        with self.summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    CellSummary(
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
        return rows

    def write_run_metadata(self, metadata: RunMetadata) -> None:
        path = self.run_dir / "run.json"
        serialized = metadata.to_json()
        if not path.exists():
            path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False))
            return

        existing = self.load_run_metadata()
        if existing == serialized:
            return

        with self.metadata_updates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serialized, ensure_ascii=False) + "\n")

    def load_run_metadata(self) -> dict[str, Any] | None:
        path = self.run_dir / "run.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

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
        for row in self.load_sample_results(path):
            if row.sample_id:
                completed.add(str(row.sample_id))
        return completed

    def load_sample_results(self, path: Path) -> list[SampleResult]:
        if not path.exists():
            return []
        rows: list[SampleResult] = []
        sample_fields = {field.name: field for field in fields(SampleResult)}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                normalized: dict[str, Any] = {}
                skip_row = False
                for name, field in sample_fields.items():
                    if name in row:
                        normalized[name] = row[name]
                        continue
                    if field.default is not MISSING:
                        normalized[name] = field.default
                        continue
                    if field.default_factory is not MISSING:  # type: ignore[comparison-overlap]
                        normalized[name] = field.default_factory()  # type: ignore[misc]
                        continue
                    log.warning("Skipping incompatible raw sample row missing required field '%s': %s", name, row)
                    skip_row = True
                    break
                if skip_row:
                    continue
                rows.append(SampleResult(**normalized))
        return rows

    def restore_accumulator(self, path: Path) -> SummaryAccumulator:
        accumulator = SummaryAccumulator()
        for result in self.load_sample_results(path):
            accumulator.update(result)
        return accumulator

    def append_sample_result(self, path: Path, result: SampleResult) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_json(), ensure_ascii=False) + "\n")

    def append_cell_summary(self, summary: CellSummary) -> None:
        key = (summary.benchmark_id, summary.runtime_label)
        replaced = False
        for index, row in enumerate(self.summary_rows):
            if (row.benchmark_id, row.runtime_label) == key:
                self.summary_rows[index] = summary
                replaced = True
                break
        if not replaced:
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
        mean_total_latency_ms=(accumulator.total_latency_sum / accumulator.total_latency_count)
        if accumulator.total_latency_count
        else 0.0,
        mean_decode_tps=(accumulator.decode_tps_sum / accumulator.decode_tps_count)
        if accumulator.decode_tps_count
        else None,
        mean_prompt_tokens=(accumulator.prompt_tokens_sum / accumulator.prompt_tokens_count)
        if accumulator.prompt_tokens_count
        else 0.0,
        mean_output_tokens=(accumulator.output_tokens_sum / accumulator.output_tokens_count)
        if accumulator.output_tokens_count
        else 0.0,
    )
