from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

from tq_bench_framework.schema import BenchmarkManifest, BenchmarkSample


class DatasetError(ValueError):
    """Raised when a benchmark dataset cannot be resolved or parsed."""


def parse_dataset_file_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise DatasetError(
                f"Dataset override '{item}' must use the form benchmark_id=/abs/path/file.jsonl"
            )
        benchmark_id, path_text = item.split("=", 1)
        overrides[benchmark_id.strip()] = Path(path_text.strip()).expanduser().resolve()
    return overrides


def resolve_dataset_file(
    manifest: BenchmarkManifest,
    overrides: dict[str, Path],
) -> Path:
    if manifest.id in overrides:
        return overrides[manifest.id]

    env_value = os.environ.get(manifest.data_file_env)
    if not env_value:
        raise DatasetError(
            f"Dataset for '{manifest.id}' is not configured. "
            f"Set {manifest.data_file_env} or pass --dataset-file {manifest.id}=/path/to/file.jsonl."
        )
    return Path(env_value).expanduser().resolve()


def _coerce_answers(record: dict, sample_id: str) -> list[str]:
    if "answers" in record:
        answers = record["answers"]
        if not isinstance(answers, list) or not answers:
            raise DatasetError(f"Sample {sample_id} has invalid 'answers'.")
        return [str(answer) for answer in answers]
    if "answer" in record:
        return [str(record["answer"])]
    raise DatasetError(f"Sample {sample_id} is missing 'answer' or 'answers'.")


def _coerce_images(record: dict, base_dir: Path) -> list[str]:
    images: list[str] = []
    if "images" in record and record["images"] is not None:
        image_values = record["images"]
        if not isinstance(image_values, list):
            raise DatasetError("'images' must be a list when present.")
        images.extend(str(item) for item in image_values)
    elif "image" in record and record["image"] is not None:
        images.append(str(record["image"]))

    resolved: list[str] = []
    for image in images:
        if image.startswith(("http://", "https://", "data:")):
            resolved.append(image)
            continue
        path = Path(image)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        resolved.append(str(path))
    return resolved


def _build_sample(
    manifest: BenchmarkManifest,
    record: dict,
    *,
    line_no: int,
    base_dir: Path,
) -> BenchmarkSample:
    sample_id = str(record.get("sample_id") or record.get("id") or f"{manifest.id}-{line_no}")
    question = str(record.get("question") or record.get("prompt") or record.get("query") or "").strip()
    if not question:
        raise DatasetError(f"Sample {sample_id} is missing a question/prompt/query field.")

    answers = _coerce_answers(record, sample_id)
    images = _coerce_images(record, base_dir)

    metadata = dict(record.get("metadata") or {})
    for key in ("sample_id", "id", "question", "prompt", "query", "answer", "answers", "image", "images", "metadata"):
        metadata.pop(key, None)

    return BenchmarkSample(
        sample_id=sample_id,
        benchmark_id=manifest.id,
        question=question,
        answers=answers,
        images=images,
        metadata=metadata,
    )


def stream_samples(
    manifest: BenchmarkManifest,
    dataset_file: Path,
) -> Iterable[BenchmarkSample]:
    base_dir = dataset_file.parent
    with dataset_file.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            yield _build_sample(manifest, record, line_no=line_no, base_dir=base_dir)


def select_samples(
    manifest: BenchmarkManifest,
    dataset_file: Path,
    *,
    num_limit: int | None,
    seed: int,
) -> list[BenchmarkSample]:
    if num_limit is None:
        return list(stream_samples(manifest, dataset_file))

    rng = random.Random(seed)
    reservoir: list[tuple[int, BenchmarkSample]] = []
    for stream_index, sample in enumerate(stream_samples(manifest, dataset_file), start=1):
        if stream_index <= num_limit:
            reservoir.append((stream_index, sample))
            continue
        swap_index = rng.randint(1, stream_index)
        if swap_index <= num_limit:
            reservoir[swap_index - 1] = (stream_index, sample)

    reservoir.sort(key=lambda item: item[0])
    return [sample for _, sample in reservoir]


def iter_selected_samples(
    manifest: BenchmarkManifest,
    dataset_file: Path,
    *,
    num_limit: int | None,
    seed: int,
) -> Iterable[BenchmarkSample]:
    if num_limit is None:
        return stream_samples(manifest, dataset_file)
    return select_samples(manifest, dataset_file, num_limit=num_limit, seed=seed)
