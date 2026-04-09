from __future__ import annotations

from pathlib import Path

from tq_bench_framework.benchmarks.registry import load_benchmark_registry
from tq_bench_framework.dataset import stream_samples
from tq_bench_framework.metrics import score_prediction
from tq_bench_framework.runner import build_prompt


def test_longbench_text_100_manifest_is_registered():
    registry = load_benchmark_registry()
    manifest = registry["longbench_text_100"]
    assert manifest.metric == "mmmu_option_match"
    assert manifest.adapter == "unified_jsonl_vqa"
    assert "{context}" in manifest.prompt_template


def test_longbench_text_100_dataset_is_text_only_and_scores():
    registry = load_benchmark_registry()
    manifest = registry["longbench_text_100"]
    dataset_file = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "processed"
        / "longbench_text_100"
        / "longbench_text_100.jsonl"
    )
    samples = list(stream_samples(manifest, dataset_file))
    assert len(samples) == 100

    sample = samples[0]
    assert sample.images == []
    prompt = build_prompt(manifest, sample.question, sample.metadata)
    assert "Context:" in prompt
    assert "Options:" in prompt

    # Option scoring should work with either the letter or the option text.
    assert score_prediction(manifest.metric, sample.answers[0], sample.answers, sample.metadata) == 1.0
    first_option = sample.metadata["options"][0]
    if sample.answers[0] == "A":
        assert score_prediction(manifest.metric, first_option, sample.answers, sample.metadata) == 1.0
