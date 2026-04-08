from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSourceSpec:
    benchmark_id: str
    repo_id: str
    split: str
    repo_config: str | None = None
    all_configs: bool = False
    question_field: str = "question"
    query_field: str | None = None
    answer_field: str | None = "answer"
    answers_field: str | None = None
    options_field: str | None = None
    image_fields: tuple[str, ...] = ("image",)
    id_field: str | None = "id"
    question_suffix: str | None = None
    note: str | None = None


DATASET_SOURCES: dict[str, DatasetSourceSpec] = {
    "textvqa": DatasetSourceSpec(
        benchmark_id="textvqa",
        repo_id="lmms-lab/textvqa",
        split="validation",
        answers_field="answers",
        image_fields=("image",),
        id_field="question_id",
        note="Community parquet mirror used because the original facebook/textvqa repo is dataset-script based.",
    ),
    "chartqa": DatasetSourceSpec(
        benchmark_id="chartqa",
        repo_id="HuggingFaceM4/ChartQA",
        split="test",
        query_field="query",
        answers_field="label",
        image_fields=("image",),
        id_field=None,
    ),
    "ai2d": DatasetSourceSpec(
        benchmark_id="ai2d",
        repo_id="lmms-lab/ai2d",
        split="test",
        options_field="options",
        image_fields=("image",),
        id_field=None,
    ),
    "docvqa": DatasetSourceSpec(
        benchmark_id="docvqa",
        repo_id="pixparse/docvqa-single-page-questions",
        split="validation",
        answers_field="answers",
        image_fields=("image",),
        id_field="question_id",
    ),
    "mathvista": DatasetSourceSpec(
        benchmark_id="mathvista",
        repo_id="AI4Math/MathVista",
        split="testmini",
        query_field="query",
        image_fields=("decoded_image",),
        id_field="pid",
    ),
    "mmmu": DatasetSourceSpec(
        benchmark_id="mmmu",
        repo_id="MMMU/MMMU",
        split="validation",
        all_configs=True,
        options_field="options",
        image_fields=("image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"),
        id_field="id",
    ),
    "ocrbench_v2": DatasetSourceSpec(
        benchmark_id="ocrbench_v2",
        repo_id="morpheushoc/OCRBenchv2",
        split="test",
        answers_field="answers",
        image_fields=("image",),
        id_field="id",
    ),
    "chartqapro": DatasetSourceSpec(
        benchmark_id="chartqapro",
        repo_id="ahmed-masry/ChartQAPro",
        split="test",
        question_field="Question",
        answer_field="Answer",
        image_fields=("image",),
        id_field=None,
        note="Public HF test split used for conditional diagnostic runs.",
    ),
}
