from __future__ import annotations

from pathlib import Path

from tq_bench_framework.benchmarks.registry import load_benchmark_registry
from tq_bench_framework.dataset import stream_samples
from tq_bench_framework.metrics import score_prediction
from tq_bench_framework.runner import build_prompt


def test_mmlu_text_1000_manifest_is_registered():
    registry = load_benchmark_registry()
    manifest = registry["mmlu_text_1000"]
    assert manifest.metric == "mmmu_option_match"
    assert manifest.adapter == "unified_jsonl_vqa"
    assert "{choice_A}" in manifest.prompt_template


def test_mmlu_text_1000_dataset_parses_and_scores():
    registry = load_benchmark_registry()
    manifest = registry["mmlu_text_1000"]
    dataset_file = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "processed"
        / "mmlu_text_1000"
        / "mmlu_text_1000.jsonl"
    )
    samples = list(stream_samples(manifest, dataset_file))
    assert len(samples) == 1000

    sample = samples[0]
    assert sample.images == []
    prompt = build_prompt(manifest, sample.question, sample.metadata)
    assert "Options:" in prompt

    assert score_prediction(manifest.metric, sample.answers[0], sample.answers, sample.metadata) == 1.0
    if sample.answers[0] == "A":
        assert score_prediction(manifest.metric, sample.metadata["options"][0], sample.answers, sample.metadata) == 1.0
