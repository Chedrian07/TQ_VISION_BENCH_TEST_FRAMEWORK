from __future__ import annotations

import json
import logging
import os
import ast
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from datasets import get_dataset_config_names, load_dataset, load_dataset_builder
from PIL import Image

from tq_bench_framework.dataset_sources import DATASET_SOURCES, DatasetSourceSpec
from tq_bench_framework.settings import FrameworkSettings

log = logging.getLogger("tq-bench.dataset")


class DatasetPreparationError(RuntimeError):
    """Raised when dataset preparation fails."""


@dataclass(frozen=True)
class DatasetCheckResult:
    benchmark_id: str
    repo_id: str
    split: str
    local_ready: bool
    local_path: str | None
    remote_ok: bool
    remote_message: str
    note: str | None = None


def available_source_ids() -> list[str]:
    return sorted(DATASET_SOURCES.keys())


def get_source(benchmark_id: str) -> DatasetSourceSpec:
    try:
        return DATASET_SOURCES[benchmark_id]
    except KeyError as exc:
        raise DatasetPreparationError(f"No dataset source registered for '{benchmark_id}'.") from exc


def _dataset_env_var(benchmark_id: str) -> str:
    return f"{benchmark_id.upper()}_DATASET_FILE"


def _prepared_jsonl_path(settings: FrameworkSettings, benchmark_id: str) -> Path:
    return settings.datasets_processed_dir / benchmark_id / f"{benchmark_id}.jsonl"


def _ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _expected_row_count(source: DatasetSourceSpec, num_limit: int | None) -> int | None:
    if source.expected_rows is None:
        return num_limit
    if num_limit is None:
        return source.expected_rows
    return min(source.expected_rows, num_limit)


def _validate_prepared_dataset(
    output_jsonl: Path,
    images_dir: Path,
    source: DatasetSourceSpec,
    num_limit: int | None,
    *,
    require_images_dir: bool = True,
) -> tuple[bool, str]:
    if not output_jsonl.exists():
        return False, "missing jsonl"
    if require_images_dir and not images_dir.exists():
        return False, "missing images directory"

    row_count = _count_jsonl_rows(output_jsonl)
    if row_count == 0:
        return False, "jsonl is empty"

    expected_rows = _expected_row_count(source, num_limit)
    if expected_rows is not None and row_count != expected_rows:
        return False, f"row count mismatch (expected {expected_rows}, found {row_count})"

    return True, f"ready ({row_count} rows)"


def _update_env_file(settings: FrameworkSettings, env_key: str, env_value: str) -> None:
    _ensure_dirs(settings.env_path)
    lines: list[str] = []
    if settings.env_path.exists():
        lines = settings.env_path.read_text(encoding="utf-8").splitlines()

    replaced = False
    for index, line in enumerate(lines):
        if line.startswith(f"{env_key}="):
            lines[index] = f'{env_key}="{env_value}"'
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f'{env_key}="{env_value}"')

    settings.env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_dataset(settings: FrameworkSettings, benchmark_id: str) -> DatasetCheckResult:
    source = get_source(benchmark_id)
    env_var = _dataset_env_var(benchmark_id)
    local_path_text = os.environ.get(env_var)
    default_local_path = _prepared_jsonl_path(settings, benchmark_id)
    local_path = Path(local_path_text).expanduser().resolve() if local_path_text else default_local_path
    local_ready, local_status = _validate_prepared_dataset(
        local_path,
        local_path.parent / "images",
        source,
        num_limit=None,
        require_images_dir=local_path == default_local_path,
    )

    remote_ok = False
    remote_message = ""
    try:
        if source.all_configs:
            builder = load_dataset_builder(
                source.repo_id,
                name=_iter_config_names(source)[0],
                cache_dir=str(settings.datasets_cache_dir),
            )
            remote_ok = True
            remote_message = (
                f"OK: split={source.split}, configs={len(_iter_config_names(source))}"
            )
        else:
            kwargs = {}
            if source.repo_config is not None:
                kwargs["name"] = source.repo_config
            builder = load_dataset_builder(
                source.repo_id,
                cache_dir=str(settings.datasets_cache_dir),
                **kwargs,
            )
            remote_ok = True
            remote_message = f"OK: splits={list(builder.info.splits.keys())}"
    except Exception as exc:  # noqa: BLE001
        remote_message = f"{type(exc).__name__}: {exc}"

    return DatasetCheckResult(
        benchmark_id=benchmark_id,
        repo_id=source.repo_id,
        split=source.split,
        local_ready=local_ready,
        local_path=str(local_path) if local_path else None,
        remote_ok=remote_ok,
        remote_message=remote_message,
        note=f"{source.note} | local_status: {local_status}" if source.note else f"local_status: {local_status}",
    )


def _extract_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        if len(value) == 1:
            return _extract_scalar(value[0])
        return "\n".join(str(item) for item in value)
    text = str(value).strip()
    return text or None


def _extract_answers(record: dict[str, Any], source: DatasetSourceSpec) -> list[str]:
    candidates: list[str] = []
    if source.answers_field and source.answers_field in record and record[source.answers_field] is not None:
        value = record[source.answers_field]
        if isinstance(value, list):
            candidates.extend(str(item).strip() for item in value if str(item).strip())
        else:
            scalar = _extract_scalar(value)
            if scalar:
                candidates.append(scalar)
    if not candidates and source.answer_field and source.answer_field in record:
        value = record[source.answer_field]
        if isinstance(value, list):
            candidates.extend(str(item).strip() for item in value if str(item).strip())
        else:
            scalar = _extract_scalar(value)
            if scalar:
                candidates.append(scalar)
    return candidates


def _extract_question(record: dict[str, Any], source: DatasetSourceSpec) -> str:
    field = source.query_field or source.question_field
    question = _extract_scalar(record.get(field)) or ""
    if not question and source.question_field != field:
        question = _extract_scalar(record.get(source.question_field)) or ""

    if source.options_field and record.get(source.options_field):
        raw_options = record[source.options_field]
        options_text = None
        if isinstance(raw_options, list):
            if raw_options and isinstance(raw_options[0], str):
                options_text = "\n".join(
                    f"{chr(65 + index)}. {value}" for index, value in enumerate(raw_options)
                )
        elif isinstance(raw_options, str):
            try:
                parsed = ast.literal_eval(raw_options)
                if isinstance(parsed, list):
                    options_text = "\n".join(
                        f"{chr(65 + index)}. {value}" for index, value in enumerate(parsed)
                    )
                else:
                    options_text = raw_options
            except (ValueError, SyntaxError):
                options_text = raw_options
        if options_text:
            question = f"{question}\nOptions:\n{options_text}".strip()

    if source.question_suffix:
        question = f"{question}\n{source.question_suffix}".strip()

    return question


def _extract_images(record: dict[str, Any], source: DatasetSourceSpec) -> list[Any]:
    images: list[Any] = []
    for field in source.image_fields:
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() == "none":
            continue
        images.append(value)
    return images


_PNG_COMPATIBLE_MODES = {"1", "L", "LA", "I", "P", "RGB", "RGBA"}
_PRESERVABLE_FORMATS: dict[str, tuple[str, str]] = {
    "PNG": (".png", "PNG"),
    "JPEG": (".jpg", "JPEG"),
    "WEBP": (".webp", "WEBP"),
    "GIF": (".gif", "GIF"),
    "BMP": (".bmp", "BMP"),
    "TIFF": (".tiff", "TIFF"),
}
_PRESERVABLE_SUFFIXES = {suffix for suffix, _ in _PRESERVABLE_FORMATS.values()} | {".jpeg", ".tif"}


def _normalized_save_target(image: Image.Image, save_format: str) -> Image.Image:
    if save_format == "JPEG":
        if image.mode not in {"L", "RGB", "CMYK"}:
            return image.convert("RGB")
        return image
    if image.mode not in _PNG_COMPATIBLE_MODES:
        target_mode = "RGBA" if "A" in image.getbands() else "RGB"
        return image.convert(target_mode)
    return image


def _resolve_save_format(
    *,
    source_path: Path | None = None,
    source_format: str | None = None,
) -> tuple[str, str]:
    if source_format:
        normalized = str(source_format).upper()
        if normalized in _PRESERVABLE_FORMATS:
            return _PRESERVABLE_FORMATS[normalized]
    if source_path is not None:
        suffix = source_path.suffix.lower()
        if suffix == ".jpeg":
            return ".jpg", "JPEG"
        if suffix == ".tif":
            return ".tiff", "TIFF"
        for known_suffix, save_format in _PRESERVABLE_FORMATS.values():
            if suffix == known_suffix:
                return known_suffix, save_format
    return ".png", "PNG"


def _save_decoded_image(
    image: Image.Image,
    output_dir: Path,
    stem: str,
    image_index: int,
    *,
    record_output_dir: Path | None = None,
    source_path: Path | None = None,
    source_format: str | None = None,
) -> str:
    suffix, save_format = _resolve_save_format(source_path=source_path, source_format=source_format or image.format)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_{image_index}{suffix}"
    recorded_path = (record_output_dir or output_dir) / f"{stem}_{image_index}{suffix}"
    normalized_image = _normalized_save_target(image, save_format)
    normalized_image.save(output_path, format=save_format)
    return str(recorded_path.resolve())


def _copy_existing_image(
    path: Path,
    output_dir: Path,
    stem: str,
    image_index: int,
    *,
    record_output_dir: Path | None = None,
) -> str | None:
    suffix = path.suffix.lower()
    if suffix not in _PRESERVABLE_SUFFIXES:
        return None
    if suffix == ".jpeg":
        suffix = ".jpg"
    elif suffix == ".tif":
        suffix = ".tiff"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_{image_index}{suffix}"
    recorded_path = (record_output_dir or output_dir) / f"{stem}_{image_index}{suffix}"
    shutil.copy2(path, output_path)
    return str(recorded_path.resolve())


def _save_image(
    image_value: Any,
    output_dir: Path,
    stem: str,
    image_index: int,
    *,
    record_output_dir: Path | None = None,
) -> str:
    if isinstance(image_value, Image.Image):
        return _save_decoded_image(
            image_value,
            output_dir,
            stem,
            image_index,
            record_output_dir=record_output_dir,
        )

    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            with Image.open(BytesIO(image_value["bytes"])) as image:
                return _save_decoded_image(
                    image,
                    output_dir,
                    stem,
                    image_index,
                    record_output_dir=record_output_dir,
                )
        if image_value.get("path"):
            path = Path(str(image_value["path"]))
            if path.exists():
                copied = _copy_existing_image(
                    path,
                    output_dir,
                    stem,
                    image_index,
                    record_output_dir=record_output_dir,
                )
                if copied is not None:
                    return copied
                with Image.open(path) as image:
                    return _save_decoded_image(
                        image,
                        output_dir,
                        stem,
                        image_index,
                        record_output_dir=record_output_dir,
                        source_path=path,
                    )

    if isinstance(image_value, (bytes, bytearray)):
        with Image.open(BytesIO(image_value)) as image:
            return _save_decoded_image(
                image,
                output_dir,
                stem,
                image_index,
                record_output_dir=record_output_dir,
            )

    if isinstance(image_value, str):
        path = Path(image_value)
        if path.exists():
            copied = _copy_existing_image(
                path,
                output_dir,
                stem,
                image_index,
                record_output_dir=record_output_dir,
            )
            if copied is not None:
                return copied
            with Image.open(path) as image:
                return _save_decoded_image(
                    image,
                    output_dir,
                    stem,
                    image_index,
                    record_output_dir=record_output_dir,
                    source_path=path,
                )

    raise DatasetPreparationError(f"Unsupported image payload type: {type(image_value).__name__}")


def _iter_config_names(source: DatasetSourceSpec) -> list[str | None]:
    if source.all_configs:
        configs = get_dataset_config_names(source.repo_id)
        return list(configs)
    return [source.repo_config]


def prepare_dataset(
    settings: FrameworkSettings,
    benchmark_id: str,
    *,
    num_limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    source = get_source(benchmark_id)
    output_jsonl = _prepared_jsonl_path(settings, benchmark_id)
    images_dir = output_jsonl.parent / "images"
    _ensure_dirs(output_jsonl)
    ready, status = _validate_prepared_dataset(output_jsonl, images_dir, source, num_limit)
    if ready and not overwrite:
        log.info("Reusing prepared dataset for %s: %s", benchmark_id, status)
        _update_env_file(settings, _dataset_env_var(benchmark_id), str(output_jsonl.resolve()))
        return output_jsonl
    if output_jsonl.exists() and not overwrite:
        log.warning(
            "Prepared dataset for %s is not reusable (%s). Regenerating it.",
            benchmark_id,
            status,
        )

    temp_jsonl = output_jsonl.parent / f".{benchmark_id}.jsonl.tmp"
    temp_images_dir = output_jsonl.parent / ".images_tmp"
    if temp_jsonl.exists():
        temp_jsonl.unlink()
    if temp_images_dir.exists():
        shutil.rmtree(temp_images_dir)

    count = 0
    try:
        with temp_jsonl.open("w", encoding="utf-8") as handle:
            for config_name in _iter_config_names(source):
                load_kwargs = {"split": source.split, "streaming": True}
                if config_name is not None:
                    load_kwargs["name"] = config_name
                dataset = load_dataset(
                    source.repo_id,
                    cache_dir=str(settings.datasets_cache_dir),
                    **load_kwargs,
                )
                for record in dataset:
                    answers = _extract_answers(record, source)
                    question = _extract_question(record, source)
                    if not answers or not question:
                        continue

                    raw_images = _extract_images(record, source)
                    sample_id = _extract_scalar(record.get(source.id_field)) if source.id_field else None
                    if not sample_id:
                        sample_id = f"{benchmark_id}-{count:08d}"
                    image_paths = [
                        _save_image(
                            image_value,
                            temp_images_dir,
                            sample_id,
                            image_index,
                            record_output_dir=images_dir,
                        )
                        for image_index, image_value in enumerate(raw_images, start=1)
                    ]

                    metadata = {
                        key: value
                        for key, value in record.items()
                        if key not in set(source.image_fields)
                        and key not in {
                            source.question_field,
                            source.query_field,
                            source.answer_field,
                            source.answers_field,
                            source.options_field,
                            source.id_field,
                        }
                    }
                    if source.options_field and record.get(source.options_field) is not None:
                        raw_options = record[source.options_field]
                        if isinstance(raw_options, list):
                            metadata["options"] = raw_options
                        elif isinstance(raw_options, str):
                            try:
                                parsed = ast.literal_eval(raw_options)
                                metadata["options"] = parsed if isinstance(parsed, list) else raw_options
                            except (ValueError, SyntaxError):
                                metadata["options"] = raw_options
                    if config_name is not None:
                        metadata["hf_config"] = config_name

                    row = {
                        "sample_id": sample_id,
                        "question": question,
                        "answers": answers,
                        "images": image_paths,
                        "metadata": metadata,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 500 == 0:
                        log.info("Prepared %d samples for %s", count, benchmark_id)
                    if num_limit is not None and count >= num_limit:
                        break
                if num_limit is not None and count >= num_limit:
                    break

        if count == 0:
            raise DatasetPreparationError(
                f"Prepared dataset for '{benchmark_id}' produced zero valid rows."
            )

        expected_rows = _expected_row_count(source, num_limit)
        if expected_rows is not None and count != expected_rows:
            raise DatasetPreparationError(
                f"Prepared dataset for '{benchmark_id}' is incomplete: expected {expected_rows} rows, found {count}."
            )

        if output_jsonl.exists():
            output_jsonl.unlink()
        if images_dir.exists():
            shutil.rmtree(images_dir)
        temp_jsonl.replace(output_jsonl)
        if temp_images_dir.exists():
            temp_images_dir.replace(images_dir)

    except Exception:
        if temp_jsonl.exists():
            temp_jsonl.unlink()
        if temp_images_dir.exists():
            shutil.rmtree(temp_images_dir)
        raise

    _update_env_file(settings, _dataset_env_var(benchmark_id), str(output_jsonl.resolve()))
    log.info("Prepared dataset for %s with %d rows -> %s", benchmark_id, count, output_jsonl)
    return output_jsonl
