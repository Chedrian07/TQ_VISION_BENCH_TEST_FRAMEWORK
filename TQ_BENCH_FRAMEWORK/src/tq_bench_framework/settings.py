from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class FrameworkSettings:
    env_files: tuple[str, ...]
    env_path: Path
    openai_api_key: str
    openai_base_url: str
    hf_token: str | None
    request_timeout_seconds: float
    reload_timeout_seconds: float
    connect_timeout_seconds: float
    max_retries: int
    datasets_root: Path
    datasets_processed_dir: Path
    datasets_cache_dir: Path
    results_dir: Path
    reports_dir: Path


def _framework_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_env_files() -> tuple[str, ...]:
    project_dir = _framework_root()
    explicit = os.environ.get("TQ_BENCH_ENV_FILE")

    candidates: list[Path] = []
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = (Path.cwd() / explicit_path).resolve()
        candidates.append(explicit_path)

    candidates.extend(
        [
            (Path.cwd() / ".env").resolve(),
            (project_dir / ".env").resolve(),
        ]
    )

    loaded: list[str] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(str(path))

    return tuple(loaded)


def _read_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _read_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _read_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def load_framework_settings() -> FrameworkSettings:
    env_files = _load_env_files()
    root = _framework_root()
    base_url = _read_str("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    env_path = (root / ".env").resolve()
    datasets_root = (root / "datasets").resolve()

    return FrameworkSettings(
        env_files=env_files,
        env_path=env_path,
        openai_api_key=_read_str("OPENAI_API_KEY", "api"),
        openai_base_url=base_url,
        hf_token=_read_str("HF_TOKEN", "").strip() or None,
        request_timeout_seconds=_read_float("TQ_BENCH_REQUEST_TIMEOUT_SECONDS", 600.0),
        reload_timeout_seconds=_read_float("TQ_BENCH_RELOAD_TIMEOUT_SECONDS", 60.0),
        connect_timeout_seconds=_read_float("TQ_BENCH_CONNECT_TIMEOUT_SECONDS", 10.0),
        max_retries=_read_int("TQ_BENCH_MAX_RETRIES", 2),
        datasets_root=datasets_root,
        datasets_processed_dir=(datasets_root / "processed").resolve(),
        datasets_cache_dir=(datasets_root / "cache").resolve(),
        results_dir=(root / "results").resolve(),
        reports_dir=(root / "reports").resolve(),
    )
