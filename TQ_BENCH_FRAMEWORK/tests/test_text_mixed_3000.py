from __future__ import annotations

from pathlib import Path

from tq_bench_framework.benchmarks.registry import load_benchmark_registry
from tq_bench_framework.dataset import stream_samples
from tq_bench_framework.metrics import score_prediction
from tq_bench_framework.runner import build_prompt


def test_text_mixed_3000_manifest_is_registered():
    registry = load_benchmark_registry()
    manifest = registry["text_mixed_3000"]
    assert manifest.metric == "mmmu_option_match"
    assert "{choices_block}" in manifest.prompt_template


def test_text_mixed_3000_dataset_parses_and_scores():
    registry = load_benchmark_registry()
    manifest = registry["text_mixed_3000"]
    dataset_file = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "processed"
        / "text_mixed_3000"
        / "text_mixed_3000.jsonl"
    )
    samples = list(stream_samples(manifest, dataset_file))
    assert len(samples) == 3000

    sample = samples[0]
    assert sample.images == []
    prompt = build_prompt(manifest, sample.question, sample.metadata)
    assert "Options:" in prompt
    assert score_prediction(manifest.metric, sample.answers[0], sample.answers, sample.metadata) == 1.0
