from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from omlx.compat.vlm import quantize_prompt_cache
from omlx.engine import VLMBatchedEngine
from omlx.model_settings import ModelSettings
from omlx.scheduler import SchedulerConfig

from tq_bench_framework.benchmarks.registry import load_benchmark_registry
from tq_bench_framework.dataset import resolve_dataset_file, stream_samples, parse_dataset_file_overrides
from tq_bench_framework.runner import build_prompt
from tq_bench_framework.settings import load_framework_settings


def _array_to_sample(arr: mx.array, max_points: int = 200_000) -> np.ndarray:
    flat = arr.reshape(-1).astype(mx.float32)
    size = int(flat.size)
    if size == 0:
        return np.array([], dtype=np.float32)
    step = max(1, size // max_points)
    sampled = flat[::step]
    return np.asarray(sampled)


def _summarize_array(arr: mx.array) -> dict[str, Any]:
    arr_f = arr.astype(mx.float32)
    sampled = _array_to_sample(arr_f)
    payload: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if int(arr.size) == 0:
        payload.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "abs_mean": None,
                "q01": None,
                "q05": None,
                "q50": None,
                "q95": None,
                "q99": None,
                "hist_edges": [],
                "hist_counts": [],
                "sample_values": [],
            }
        )
        return payload

    payload.update(
        {
            "min": float(mx.min(arr_f).item()),
            "max": float(mx.max(arr_f).item()),
            "mean": float(mx.mean(arr_f).item()),
            "std": float(mx.std(arr_f).item()),
            "abs_mean": float(mx.mean(mx.abs(arr_f)).item()),
            "sample_size": int(sampled.size),
            "sample_values": sampled[:64].astype(float).tolist(),
        }
    )

    if sampled.size:
        payload.update(
            {
                "q01": float(np.quantile(sampled, 0.01)),
                "q05": float(np.quantile(sampled, 0.05)),
                "q50": float(np.quantile(sampled, 0.50)),
                "q95": float(np.quantile(sampled, 0.95)),
                "q99": float(np.quantile(sampled, 0.99)),
            }
        )
        hist_counts, hist_edges = np.histogram(sampled, bins=32)
        payload["hist_edges"] = hist_edges.astype(float).tolist()
        payload["hist_counts"] = hist_counts.astype(int).tolist()
    else:
        payload.update(
            {
                "q01": None,
                "q05": None,
                "q50": None,
                "q95": None,
                "q99": None,
                "hist_edges": [],
                "hist_counts": [],
            }
        )

    return payload


def _describe_state(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, mx.array):
        return _summarize_array(obj)
    if hasattr(obj, "_state"):  # _QuantizedStateProxy
        return _describe_state(obj._state)
    if hasattr(obj, "_fields"):  # namedtuple-like state
        return {field: _describe_state(getattr(obj, field)) for field in obj._fields}
    if isinstance(obj, dict):
        return {str(k): _describe_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_describe_state(v) for v in obj]
    return repr(obj)


def _collect_samples(layer_payloads: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    samples: list[float] = []
    for layer in layer_payloads:
        section = layer.get(key)
        if not section:
            continue
        values = section.get("sample_values") or []
        samples.extend(values)
    if not samples:
        return None
    arr = np.asarray(samples, dtype=np.float32)
    hist_counts, hist_edges = np.histogram(arr, bins=32)
    return {
        "sample_size": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "hist_edges": hist_edges.astype(float).tolist(),
        "hist_counts": hist_counts.astype(int).tolist(),
    }


def _layer_payload(layer_index: int, cache_obj: Any) -> dict[str, Any]:
    cache = getattr(cache_obj, "_cache", cache_obj)
    payload: dict[str, Any] = {
        "layer_index": layer_index,
        "cache_class": type(cache).__name__,
    }
    state = getattr(cache, "state", None)
    if state is None:
        payload["state"] = None
        return payload

    state_value = state() if callable(state) else state
    payload["raw_state"] = _describe_state(state_value)

    if hasattr(cache, "dequantize"):
        try:
            keys, values = cache.dequantize()
            payload["dequantized_keys"] = _summarize_array(keys)
            payload["dequantized_values"] = _summarize_array(values)
            payload["token_count"] = int(keys.shape[2]) if keys.ndim >= 3 else None
        except Exception as exc:  # noqa: BLE001
            payload["dequantize_error"] = repr(exc)
        return payload

    if isinstance(state_value, tuple) and len(state_value) >= 2:
        keys, values = state_value[:2]
        if isinstance(keys, mx.array) and isinstance(values, mx.array):
            payload["keys"] = _summarize_array(keys)
            payload["values"] = _summarize_array(values)
            payload["token_count"] = int(keys.shape[2]) if keys.ndim >= 3 else None

    return payload


def _build_messages(system_prompt: str | None, prompt: str, images: list[str]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if images:
        content = [{"type": "text", "text": prompt}] + [
            {"type": "input_image", "image_url": image} for image in images
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


@dataclass
class Args:
    benchmark: str
    sample_index: int | None
    sample_id: str | None
    scheme: str
    bits: float | None
    output: Path
    dataset_file_overrides: list[str]
    model: str
    trust_remote_code: bool
    prefill_step_size: int


async def _dump(args: Args) -> dict[str, Any]:
    # Load .env / framework settings so dataset env vars behave the same way as tq-bench run.
    load_framework_settings()

    registry = load_benchmark_registry()
    manifest = registry[args.benchmark]
    overrides = parse_dataset_file_overrides(args.dataset_file_overrides)
    dataset_file = resolve_dataset_file(manifest, overrides)
    samples = list(stream_samples(manifest, dataset_file))

    if args.sample_id is not None:
        sample = next(sample for sample in samples if sample.sample_id == args.sample_id)
    else:
        if args.sample_index is None:
            raise ValueError("Either --sample-index or --sample-id must be provided.")
        sample = samples[args.sample_index]

    prompt = build_prompt(manifest, sample.question, sample.metadata)
    messages = _build_messages(manifest.system_prompt, prompt, sample.images)

    model_settings = None
    if args.scheme == "turboquant" and args.bits is not None:
        model_settings = ModelSettings()
        model_settings.turboquant_kv_enabled = True
        model_settings.turboquant_kv_bits = float(args.bits)

    engine = VLMBatchedEngine(
        model_name=args.model,
        trust_remote_code=args.trust_remote_code,
        scheduler_config=SchedulerConfig(),
        model_settings=model_settings,
    )
    await engine.start()
    try:
        prompt_or_tokens, vlm_embeds, vlm_kwargs, _ = engine._process_chat_messages(
            messages, None, {}
        )

        adapter = engine._adapter
        if adapter is None:
            raise RuntimeError("Engine adapter not initialized.")

        prompt_cache = adapter.make_cache()

        if isinstance(prompt_or_tokens, list):
            input_ids = mx.array([prompt_or_tokens])
        else:
            token_ids = engine.tokenizer.encode(prompt_or_tokens)
            input_ids = mx.array([token_ids])
        full_prompt_tokens = int(input_ids.shape[1])

        if vlm_embeds is not None:
            kwargs = dict(vlm_kwargs or {})
            inputs_embeds = vlm_embeds
            while inputs_embeds.shape[1] > 1:
                n_to_process = min(args.prefill_step_size, inputs_embeds.shape[1] - 1)
                adapter(
                    input_ids[:, :n_to_process],
                    cache=prompt_cache,
                    inputs_embeds=inputs_embeds[:, :n_to_process],
                    n_to_process=n_to_process,
                    **kwargs,
                )
                quantize_prompt_cache(
                    prompt_cache,
                    quantized_kv_start=0,
                    kv_group_size=64,
                    kv_bits=args.bits,
                    kv_quant_scheme=args.scheme,
                )
                mx.eval([c.state for c in prompt_cache])
                inputs_embeds = inputs_embeds[:, n_to_process:]
                input_ids = input_ids[:, n_to_process:]
        else:
            if input_ids.shape[1] > 1:
                n_to_process = input_ids.shape[1] - 1
                adapter(
                    input_ids[:, :n_to_process],
                    cache=prompt_cache,
                    n_to_process=n_to_process,
                )
                quantize_prompt_cache(
                    prompt_cache,
                    quantized_kv_start=0,
                    kv_group_size=64,
                    kv_bits=args.bits,
                    kv_quant_scheme=args.scheme,
                )
                mx.eval([c.state for c in prompt_cache])

        layer_payloads = [_layer_payload(i, cache) for i, cache in enumerate(prompt_cache)]

        aggregate = {
            "keys": _collect_samples(layer_payloads, "keys")
            or _collect_samples(layer_payloads, "dequantized_keys"),
            "values": _collect_samples(layer_payloads, "values")
            or _collect_samples(layer_payloads, "dequantized_values"),
        }
        quantized_layer_count = sum(
            1
            for layer in layer_payloads
            if "dequantized_keys" in layer
            or layer.get("cache_class") in {
                "QuantizedKVCache",
                "BatchQuantizedKVCache",
                "TurboQuantKVCache",
                "BatchTurboQuantKVCache",
            }
        )

        return {
            "benchmark": manifest.id,
            "sample_id": sample.sample_id,
            "question": sample.question,
            "images": sample.images,
            "runtime": {
                "scheme": args.scheme,
                "bits": args.bits,
                "model": args.model,
            },
            "prompt_tokens": full_prompt_tokens,
            "prefill_tokens": max(full_prompt_tokens - 1, 0),
            "quantized_layer_count": quantized_layer_count,
            "layers": layer_payloads,
            "aggregate": aggregate,
        }
    finally:
        await engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--scheme", choices=["none", "mlx", "turboquant"], default="none")
    parser.add_argument("--bits", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-file", action="append", default=[])
    parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-MLX-bf16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    ns = parser.parse_args()

    args = Args(
        benchmark=ns.benchmark,
        sample_index=ns.sample_index,
        sample_id=ns.sample_id,
        scheme=ns.scheme,
        bits=ns.bits,
        output=Path(ns.output).expanduser().resolve(),
        dataset_file_overrides=ns.dataset_file,
        model=ns.model,
        trust_remote_code=ns.trust_remote_code,
        prefill_step_size=ns.prefill_step_size,
    )

    payload = asyncio.run(_dump(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
