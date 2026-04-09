from __future__ import annotations

from pathlib import Path

from tq_bench_framework.benchmarks.registry import load_benchmark_registry
from tq_bench_framework.dataset import stream_samples
from tq_bench_framework.metrics import score_prediction
from tq_bench_framework.runner import build_prompt


def _check_dataset(benchmark_id: str, rel_path: str) -> None:
    registry = load_benchmark_registry()
    manifest = registry[benchmark_id]
    dataset_file = Path(__file__).resolve().parents[1] / rel_path
    samples = list(stream_samples(manifest, dataset_file))
    assert len(samples) == 3000
    sample = samples[0]
    assert sample.images == []
    prompt = build_prompt(manifest, sample.question, sample.metadata)
    assert "Options:" in prompt
    assert score_prediction(manifest.metric, sample.answers[0], sample.answers, sample.metadata) == 1.0


def test_mmlu_text_3000_dataset():
    _check_dataset(
        "mmlu_text_3000",
        "datasets/processed/mmlu_text_3000/mmlu_text_3000.jsonl",
    )


def test_commonsenseqa_text_3000_dataset():
    _check_dataset(
        "commonsenseqa_text_3000",
        "datasets/processed/commonsenseqa_text_3000/commonsenseqa_text_3000.jsonl",
    )


def test_hellaswag_text_3000_dataset():
    _check_dataset(
        "hellaswag_text_3000",
        "datasets/processed/hellaswag_text_3000/hellaswag_text_3000.jsonl",
    )
