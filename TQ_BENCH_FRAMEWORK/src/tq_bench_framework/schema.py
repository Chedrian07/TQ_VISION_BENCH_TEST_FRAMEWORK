from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkManifest:
    id: str
    title: str
    stage: str
    description: str
    data_file_env: str
    adapter: str
    metric: str
    default_sampling_profile: str
    best_effort_sampling_profile: str
    max_output_tokens: int
    prompt_template: str = "{question}"
    system_prompt: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    benchmark_id: str
    question: str
    answers: list[str]
    images: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeConfig:
    scheme: str
    bits: float | None
    sampling_profile: str
    model: str | None = None
    revision: str | None = None
    adapter_path: str | None = None

    @property
    def quant_label(self) -> str:
        if self.scheme == "none":
            return "baseline"
        bits = int(self.bits) if self.bits is not None and float(self.bits).is_integer() else self.bits
        prefix = "mlx" if self.scheme == "mlx" else "tq"
        return f"{prefix}-{bits}"

    @property
    def label(self) -> str:
        return f"{self.quant_label}__{self.sampling_profile}"

    @property
    def filename_label(self) -> str:
        label = self.label.replace(".", "_")
        return label.replace("/", "_")

    def reload_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kv_quant_scheme": self.scheme,
            "sampling_profile": self.sampling_profile,
        }
        if self.bits is not None:
            payload["kv_bits"] = self.bits
        if self.model is not None:
            payload["model"] = self.model
        if self.revision is not None:
            payload["revision"] = self.revision
        if self.adapter_path is not None:
            payload["adapter_path"] = self.adapter_path
        return payload


@dataclass(frozen=True)
class SampleResult:
    run_id: str
    benchmark_id: str
    benchmark_title: str
    sample_id: str
    runtime_label: str
    quant_scheme: str
    quant_bits: float | None
    sampling_profile: str
    question: str
    answers: list[str]
    prediction: str
    score: float
    metric: str
    ttft_ms: float | None
    total_latency_ms: float
    decode_latency_ms: float | None
    decode_tps: float | None
    prompt_tokens: int
    output_tokens: int
    images: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CellSummary:
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

    def to_csv_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    run_dir: Path
    selected_benchmarks: list[str]
    runtime_matrix: list[str]
    num_limit: int | None
    seed: int
    sampling_profile_mode: str
    resumed_from_run_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        return payload
