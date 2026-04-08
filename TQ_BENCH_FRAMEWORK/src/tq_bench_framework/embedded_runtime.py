"""In-process oMLX VLM runtime that bypasses the HTTP backend.

This module embeds an :class:`omlx.engine.VLMBatchedEngine` directly
inside the bench framework process, providing the same semantics as
``TQ_BACKEND/serve.py``'s ``Runtime`` class but without the
HTTP/uvicorn/SSE serialization layers in between. The savings on
short-prompt benchmarks (where vision encode + per-request fixed
overhead dominate) are substantial — typically 30-80 ms per request.

Architecture
------------
- A dedicated background thread owns an asyncio event loop and the
  oMLX engine. All MLX work happens on that loop's mlx_executor (oMLX
  internal), so we keep oMLX's threading invariants intact.
- ``EmbeddedRuntime`` exposes synchronous methods (``stream_response``,
  ``reload``) by submitting coroutines to the background loop and
  blocking on the resulting ``concurrent.futures.Future``.
- Reload is a full unload + load (mirrors TQ_BACKEND.Runtime.reconfigure).
- Vision feature SSD cache is enabled via ``paged_ssd_cache_dir`` in
  the scheduler config so cached features persist across reloads.

This module imports oMLX lazily so that pure-HTTP bench runs do not
pay the import cost.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("tq-bench.embedded")


@dataclass
class EmbeddedRuntimeSettings:
    """Subset of ``TQ_BACKEND.ServerSettings`` needed for in-process inference.

    Loaded from environment variables so the in-process path stays
    config-compatible with the existing ``TQ_BACKEND/.env`` file.
    """

    model_id: str
    adapter_path: Optional[str]
    revision: Optional[str]
    trust_remote_code: bool
    force_disable_thinking: bool
    sampling_profile: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: Optional[float]
    repetition_penalty: Optional[float]
    default_max_output_tokens: int
    default_prefill_step_size: int
    max_concurrent_requests: int
    kv_quant_scheme: str
    kv_bits: Optional[float]
    kv_group_size: int
    quantized_kv_start: int
    paged_ssd_cache_dir: Optional[str]

    @staticmethod
    def from_env() -> "EmbeddedRuntimeSettings":
        def _get(name: str, default: str = "") -> str:
            return (os.environ.get(name) or default).strip()

        def _opt(name: str) -> Optional[str]:
            v = _get(name)
            return v or None

        def _bool(name: str, default: bool) -> bool:
            v = _get(name)
            if not v:
                return default
            return v.lower() in {"1", "true", "yes", "on"}

        def _int(name: str, default: int) -> int:
            v = _get(name)
            return int(v) if v else default

        def _float(name: str, default: float) -> float:
            v = _get(name)
            return float(v) if v else default

        def _opt_float(name: str) -> Optional[float]:
            v = _get(name)
            return float(v) if v else None

        profile = _get("TQ_SAMPLING_PROFILE", "controlled")

        # Read profile-specific sampling defaults; fallback to controlled
        prefix_map = {
            "controlled": "TQ_CONTROLLED",
            "best_effort_general": "TQ_BEST_EFFORT_GENERAL",
            "best_effort_reasoning": "TQ_BEST_EFFORT_REASONING",
        }
        prefix = prefix_map.get(profile, "TQ_CONTROLLED")
        defaults_for_profile = {
            "controlled": (0.0, 1.0, 0, 0.0, None, None),
            "best_effort_general": (0.7, 0.8, 20, 0.0, 1.5, None),
            "best_effort_reasoning": (1.0, 0.95, 20, 0.0, 1.5, None),
        }[profile if profile in prefix_map else "controlled"]
        temp = _float(f"{prefix}_TEMPERATURE", defaults_for_profile[0])
        top_p = _float(f"{prefix}_TOP_P", defaults_for_profile[1])
        top_k = _int(f"{prefix}_TOP_K", defaults_for_profile[2])
        min_p = _float(f"{prefix}_MIN_P", defaults_for_profile[3])
        pres = _opt_float(f"{prefix}_PRESENCE_PENALTY")
        if pres is None:
            pres = defaults_for_profile[4]
        rep = _opt_float(f"{prefix}_REPETITION_PENALTY")
        if rep is None:
            rep = defaults_for_profile[5]

        # Mirror TQ_BACKEND._resolve_paged_ssd_cache_dir
        cache_raw = os.environ.get("TQ_OMLX_CACHE_DIR")
        if cache_raw is None:
            cache_dir: Optional[str] = str((Path.home() / ".omlx_bench" / "cache").resolve())
        else:
            cache_raw = cache_raw.strip()
            cache_dir = str(Path(cache_raw).expanduser().resolve()) if cache_raw else None

        return EmbeddedRuntimeSettings(
            model_id=_get("TQ_MODEL", "mlx-community/Qwen3.5-0.8B-MLX-bf16"),
            adapter_path=_opt("TQ_ADAPTER_PATH"),
            revision=_opt("TQ_REVISION"),
            trust_remote_code=_bool("TQ_TRUST_REMOTE_CODE", False),
            force_disable_thinking=_bool("TQ_FORCE_DISABLE_THINKING", True),
            sampling_profile=profile or "controlled",
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=pres,
            repetition_penalty=rep,
            default_max_output_tokens=_int("TQ_DEFAULT_MAX_OUTPUT_TOKENS", 512),
            default_prefill_step_size=_int("TQ_PREFILL_STEP_SIZE", 2048),
            max_concurrent_requests=_int("TQ_MAX_CONCURRENT_REQUESTS", 1),
            kv_quant_scheme=_get("TQ_KV_QUANT_SCHEME", "turboquant"),
            kv_bits=_opt_float("TQ_KV_BITS"),
            kv_group_size=_int("TQ_KV_GROUP_SIZE", 64),
            quantized_kv_start=_int("TQ_QUANTIZED_KV_START", 0),
            paged_ssd_cache_dir=cache_dir,
        )

    @property
    def active_kv_quant_scheme(self) -> Optional[str]:
        if self.kv_quant_scheme == "none":
            return None
        if self.kv_quant_scheme == "mlx":
            return "uniform"
        return self.kv_quant_scheme

    def public_state(self, *, loaded: bool) -> dict[str, Any]:
        return {
            "loaded": loaded,
            "backend": "embedded",
            "model": {
                "id": self.model_id,
                "adapter_path": self.adapter_path,
                "revision": self.revision,
            },
            "runtime": {
                "max_concurrent_requests": self.max_concurrent_requests,
            },
            "sampling_profile": self.sampling_profile,
            "sampling": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "presence_penalty": self.presence_penalty,
                "repetition_penalty": self.repetition_penalty,
                "max_output_tokens": self.default_max_output_tokens,
            },
            "thinking": {
                "force_disable_thinking": self.force_disable_thinking,
            },
            "kv_cache": {
                "scheme": self.kv_quant_scheme,
                "mlx_vlm_scheme": self.active_kv_quant_scheme,
                "bits": self.kv_bits,
                "group_size": self.kv_group_size,
                "quantized_kv_start": self.quantized_kv_start,
                "turboquant_seed": 0,
                "turboquant_from_first_token": True,
            },
            "paged_ssd_cache_dir": self.paged_ssd_cache_dir,
        }


def _profile_defaults(profile: str) -> tuple[float, float, int, float, Optional[float], Optional[float]]:
    return {
        "controlled": (0.0, 1.0, 0, 0.0, None, None),
        "best_effort_general": (0.7, 0.8, 20, 0.0, 1.5, None),
        "best_effort_reasoning": (1.0, 0.95, 20, 0.0, 1.5, None),
    }[profile]


def _replace_for_profile(
    base: EmbeddedRuntimeSettings, profile: str
) -> EmbeddedRuntimeSettings:
    """When the bench framework asks for a profile change, swap the active
    sampling tuple. We mirror the env-based defaults rather than reading
    new env vars at reload time so the in-process path stays
    deterministic per process lifetime.
    """
    defaults = _profile_defaults(profile)
    return replace(
        base,
        sampling_profile=profile,
        temperature=defaults[0],
        top_p=defaults[1],
        top_k=defaults[2],
        min_p=defaults[3],
        presence_penalty=defaults[4],
        repetition_penalty=defaults[5],
    )


class EmbeddedRuntime:
    """Owns a single in-process oMLX ``VLMBatchedEngine`` and a background
    asyncio loop to drive it.

    Public API mirrors what the bench framework needs from
    ``BackendClient``: ``state``, ``reload``, ``generate``.
    """

    def __init__(self, settings: EmbeddedRuntimeSettings):
        self.settings = settings
        self._engine: Any = None  # omlx.engine.VLMBatchedEngine
        self._engine_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._start_loop()

    # ------------------------------------------------------------------
    # Background event loop
    # ------------------------------------------------------------------

    def _start_loop(self) -> None:
        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        self._loop_thread = threading.Thread(
            target=_runner, name="embedded-omlx-loop", daemon=True
        )
        self._loop_thread.start()
        self._loop_ready.wait(timeout=10.0)
        if self._loop is None:
            raise RuntimeError("Embedded runtime: background loop failed to start")

    def _submit(self, coro):
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._submit(self._engine.stop()).result(timeout=60.0)
            except Exception:  # noqa: BLE001
                log.exception("Embedded runtime: engine stop failed")
            self._engine = None
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def _build_engine(self) -> Any:
        from omlx.engine import VLMBatchedEngine
        from omlx.model_settings import ModelSettings
        from omlx.scheduler import SchedulerConfig as OMLXSchedulerConfig
        from mlx_vlm.utils import get_model_path

        cache_dir = self.settings.paged_ssd_cache_dir
        if cache_dir:
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                log.warning(
                    "Embedded runtime: cannot create cache_dir=%s (%s); disabling SSD cache",
                    cache_dir,
                    exc,
                )
                cache_dir = None

        scheduler_cfg = OMLXSchedulerConfig(
            max_num_seqs=self.settings.max_concurrent_requests,
            completion_batch_size=self.settings.max_concurrent_requests,
            prefill_step_size=self.settings.default_prefill_step_size,
            model_name=self.settings.model_id,
            paged_ssd_cache_dir=cache_dir,
        )

        model_settings = ModelSettings()
        if (
            self.settings.kv_quant_scheme == "turboquant"
            and self.settings.kv_bits is not None
        ):
            model_settings.turboquant_kv_enabled = True
            model_settings.turboquant_kv_bits = float(self.settings.kv_bits)
        elif (
            self.settings.kv_quant_scheme == "mlx"
            and self.settings.kv_bits is not None
        ):
            model_settings.uniform_kv_enabled = True
            model_settings.uniform_kv_bits = int(self.settings.kv_bits)
            model_settings.uniform_kv_group_size = self.settings.kv_group_size
            model_settings.uniform_quantized_kv_start = self.settings.quantized_kv_start

        resolved_path = str(
            get_model_path(self.settings.model_id, revision=self.settings.revision)
        )

        engine = VLMBatchedEngine(
            model_name=resolved_path,
            trust_remote_code=self.settings.trust_remote_code,
            scheduler_config=scheduler_cfg,
            enable_thinking=not self.settings.force_disable_thinking,
            model_settings=model_settings,
        )
        return engine

    def ensure_loaded(self) -> None:
        with self._engine_lock:
            if self._engine is not None:
                return
            log.info(
                "Embedded runtime: loading model %s (kv_scheme=%s kv_bits=%s profile=%s)",
                self.settings.model_id,
                self.settings.kv_quant_scheme,
                self.settings.kv_bits,
                self.settings.sampling_profile,
            )
            engine = self._build_engine()
            self._submit(engine.start()).result()
            self._engine = engine

    def reload(
        self,
        *,
        kv_quant_scheme: Optional[str] = None,
        kv_bits: Optional[float] = None,
        sampling_profile: Optional[str] = None,
        model_id: Optional[str] = None,
        adapter_path: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._engine_lock:
            previous_state = self.settings.public_state(loaded=self._engine is not None)
            new_settings = self.settings
            if kv_quant_scheme is not None:
                # canonicalize
                normalized = kv_quant_scheme.strip().lower()
                aliases = {
                    "none": "none",
                    "off": "none",
                    "mlx": "mlx",
                    "default": "mlx",
                    "uniform": "mlx",
                    "turboquant": "turboquant",
                }
                if normalized not in aliases:
                    raise ValueError(
                        f"Embedded runtime: unsupported kv_quant_scheme={kv_quant_scheme!r}"
                    )
                normalized = aliases[normalized]
                new_settings = replace(
                    new_settings,
                    kv_quant_scheme=normalized,
                    kv_bits=kv_bits if kv_bits is not None else None
                    if normalized == "none"
                    else (kv_bits if kv_bits is not None else new_settings.kv_bits),
                )
            elif kv_bits is not None:
                new_settings = replace(new_settings, kv_bits=kv_bits)

            if sampling_profile is not None:
                new_settings = _replace_for_profile(new_settings, sampling_profile)

            if model_id is not None:
                new_settings = replace(new_settings, model_id=model_id)
            if adapter_path is not None:
                new_settings = replace(new_settings, adapter_path=adapter_path)
            if revision is not None:
                new_settings = replace(new_settings, revision=revision)

            # Tear down + rebuild
            if self._engine is not None:
                try:
                    self._submit(self._engine.stop()).result(timeout=120.0)
                except Exception:  # noqa: BLE001
                    log.exception("Embedded runtime: stop during reload failed")
                self._engine = None

            self.settings = new_settings
            engine = self._build_engine()
            self._submit(engine.start()).result()
            self._engine = engine
            new_state = self.settings.public_state(loaded=True)
            return {
                "object": "runtime.reload",
                "previous": previous_state,
                "current": new_state,
            }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def stream_response_sync(
        self,
        *,
        prompt: str,
        images: list[str],
        max_output_tokens: int,
        system_prompt: Optional[str],
    ) -> dict[str, Any]:
        """Run a single VLM request through the embedded engine and return
        the same dict shape as ``BackendClient.stream_response``.
        """
        self.ensure_loaded()
        future = self._submit(
            self._stream_async(
                prompt=prompt,
                images=images,
                max_output_tokens=max_output_tokens,
                system_prompt=system_prompt,
            )
        )
        return future.result()

    async def _stream_async(
        self,
        *,
        prompt: str,
        images: list[str],
        max_output_tokens: int,
        system_prompt: Optional[str],
    ) -> dict[str, Any]:
        if self._engine is None:
            raise RuntimeError("Embedded runtime: engine not loaded")

        # Build the messages structure that VLMBatchedEngine.stream_chat
        # expects. Mirrors the backend's HTTP path
        # (TQ_BACKEND/serve.py::build_messages): text parts + image
        # parts where each image uses ``{"type": "input_image",
        # "image_url": <url>}``. omlx ``extract_images_from_messages``
        # accepts both ``image_url`` and ``input_image`` part types and
        # the image_url field can be a plain string. The earlier
        # ``{"type": "image", "image": ...}`` shape was wrong and
        # silently dropped images, which made the model answer
        # text-only and produced lower scores than the HTTP path.
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content_parts.append({"type": "input_image", "image_url": image})

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content_parts})

        t_start = time.perf_counter()
        first_delta_ts: float | None = None
        deltas: list[str] = []
        last_output: Any = None

        async for output in self._engine.stream_chat(
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            top_k=self.settings.top_k,
            min_p=self.settings.min_p,
            repetition_penalty=self.settings.repetition_penalty or 1.0,
            presence_penalty=self.settings.presence_penalty or 0.0,
        ):
            # Use new_text strictly (matches the dedup fix in TQ_BACKEND/serve.py).
            delta = output.new_text or ""
            if delta and first_delta_ts is None:
                first_delta_ts = time.perf_counter()
            if delta:
                deltas.append(delta)
            last_output = output

        t_done = time.perf_counter()
        if last_output is None:
            raise RuntimeError("Embedded runtime: stream produced no output")

        full_text = "".join(deltas) or (getattr(last_output, "text", "") or "")
        prompt_tokens = int(getattr(last_output, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(last_output, "completion_tokens", 0) or 0)

        total_latency_ms = (t_done - t_start) * 1000.0
        ttft_ms = None if first_delta_ts is None else (first_delta_ts - t_start) * 1000.0
        decode_latency_ms = (
            None if first_delta_ts is None else (t_done - first_delta_ts) * 1000.0
        )
        decode_tps = None
        if decode_latency_ms and decode_latency_ms > 0 and output_tokens > 0:
            decode_tps = output_tokens / (decode_latency_ms / 1000.0)

        return {
            "response": {
                "id": f"resp_{uuid.uuid4().hex[:24]}",
                "object": "response",
                "status": "completed",
                "model": self.settings.model_id,
                "output_text": full_text,
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": prompt_tokens + output_tokens,
                },
            },
            "output_text": full_text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "total_latency_ms": total_latency_ms,
            "decode_latency_ms": decode_latency_ms,
            "decode_tps": decode_tps,
        }
