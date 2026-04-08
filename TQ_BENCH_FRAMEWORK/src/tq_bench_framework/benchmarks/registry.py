from __future__ import annotations

from pathlib import Path

import yaml

from tq_bench_framework.schema import BenchmarkManifest


def manifests_dir() -> Path:
    return Path(__file__).resolve().parent / "manifests"


def load_benchmark_registry() -> dict[str, BenchmarkManifest]:
    registry: dict[str, BenchmarkManifest] = {}
    for path in sorted(manifests_dir().glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        manifest = BenchmarkManifest(**raw)
        registry[manifest.id] = manifest
    return registry
