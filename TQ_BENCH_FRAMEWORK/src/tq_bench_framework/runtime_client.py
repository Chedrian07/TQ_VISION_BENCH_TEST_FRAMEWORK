from __future__ import annotations

import json
import time
from typing import Any, Protocol

import httpx

from tq_bench_framework.schema import RuntimeConfig
from tq_bench_framework.settings import FrameworkSettings


class BackendClientProtocol(Protocol):
    """Common interface for HTTP and embedded backend clients."""

    def close(self) -> None: ...
    def get_runtime(self) -> dict[str, Any]: ...
    def reload_runtime(self, config: RuntimeConfig) -> dict[str, Any]: ...
    def list_models(self) -> dict[str, Any]: ...
    def stream_response(
        self,
        *,
        model: str,
        prompt: str,
        images: list[str],
        max_output_tokens: int,
        system_prompt: str | None = None,
    ) -> dict[str, Any]: ...


class BackendClient:
    def __init__(self, settings: FrameworkSettings):
        self.settings = settings
        request_timeout = httpx.Timeout(
            settings.request_timeout_seconds,
            connect=settings.connect_timeout_seconds,
        )
        control_timeout = httpx.Timeout(
            settings.reload_timeout_seconds,
            connect=settings.connect_timeout_seconds,
        )
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.Client(
            timeout=request_timeout,
            headers=headers,
        )
        self.control_client = httpx.Client(
            timeout=control_timeout,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self.client.close()
        self.control_client.close()

    def _url(self, path: str) -> str:
        return f"{self.settings.openai_base_url}{path}"

    def get_runtime(self) -> dict[str, Any]:
        response = self.control_client.get(self._url("/runtime"))
        response.raise_for_status()
        return response.json()

    def reload_runtime(self, config: RuntimeConfig) -> dict[str, Any]:
        response = self.control_client.post(self._url("/runtime/reload"), json=config.reload_payload())
        response.raise_for_status()
        return response.json()

    def list_models(self) -> dict[str, Any]:
        response = self.control_client.get(self._url("/models"))
        response.raise_for_status()
        return response.json()

    def stream_response(
        self,
        *,
        model: str,
        prompt: str,
        images: list[str],
        max_output_tokens: int,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "stream": True,
            "max_output_tokens": max_output_tokens,
            "input": [
                {
                    "role": "user",
                    "content": (
                        [{"type": "input_text", "text": prompt}]
                        + [{"type": "input_image", "image_url": image} for image in images]
                    ),
                }
            ],
        }
        if system_prompt:
            payload["instructions"] = system_prompt

        attempt = 0
        while True:
            try:
                return self._stream_once(payload)
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > self.settings.max_retries or not self._is_retryable_exception(exc):
                    raise exc
                time.sleep(0.5 * (2 ** (attempt - 1)))

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True
        if isinstance(exc, json.JSONDecodeError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            return status_code >= 500 or status_code == 429
        return False

    def _stream_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        t_request_start = time.perf_counter()
        current_event: str | None = None
        deltas: list[str] = []
        completed_response: dict[str, Any] | None = None
        first_delta_ts: float | None = None

        with self.client.stream("POST", self._url("/responses"), json=payload) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = line.split(":", 1)[1].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    continue
                event_payload = json.loads(data)
                if current_event == "response.output_text.delta":
                    if first_delta_ts is None:
                        first_delta_ts = time.perf_counter()
                    deltas.append(str(event_payload.get("delta", "")))
                elif current_event == "response.failed":
                    error = event_payload.get("response", {}).get("error", {})
                    raise RuntimeError(error.get("message", "Backend streaming failed"))
                elif current_event == "response.completed":
                    completed_response = event_payload.get("response")

        t_done = time.perf_counter()
        if completed_response is None:
            raise RuntimeError("Backend stream finished without response.completed event.")

        output_text = str(completed_response.get("output_text") or "".join(deltas))
        usage = completed_response.get("usage") or {}
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        prompt_tokens = int(usage.get("input_tokens", 0) or 0)

        total_latency_ms = (t_done - t_request_start) * 1000.0
        ttft_ms = None if first_delta_ts is None else (first_delta_ts - t_request_start) * 1000.0
        decode_latency_ms = None if first_delta_ts is None else (t_done - first_delta_ts) * 1000.0
        decode_tps = None
        if decode_latency_ms and decode_latency_ms > 0 and output_tokens > 0:
            decode_tps = output_tokens / (decode_latency_ms / 1000.0)

        return {
            "response": completed_response,
            "output_text": output_text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "total_latency_ms": total_latency_ms,
            "decode_latency_ms": decode_latency_ms,
            "decode_tps": decode_tps,
        }


class EmbeddedBackendClient:
    """In-process backend client that drives oMLX directly via
    :class:`tq_bench_framework.embedded_runtime.EmbeddedRuntime`.

    The HTTP+SSE+uvicorn layer is bypassed entirely, saving ~30-80 ms
    per request on top of the savings from the SSD vision feature
    cache. Useful when the bench framework and oMLX run on the same
    machine and the HTTP server provides no value (purely benchmark
    workloads).
    """

    def __init__(self, settings: FrameworkSettings):
        from tq_bench_framework.embedded_runtime import (
            EmbeddedRuntime,
            EmbeddedRuntimeSettings,
        )

        self.settings = settings
        rt_settings = EmbeddedRuntimeSettings.from_env()
        self._runtime = EmbeddedRuntime(rt_settings)
        # Eager-load so the first sample doesn't pay the model load cost.
        self._runtime.ensure_loaded()

    def close(self) -> None:
        self._runtime.close()

    def get_runtime(self) -> dict[str, Any]:
        return {
            "object": "runtime",
            **self._runtime.settings.public_state(loaded=self._runtime._engine is not None),
        }

    def reload_runtime(self, config: RuntimeConfig) -> dict[str, Any]:
        return self._runtime.reload(
            kv_quant_scheme=config.scheme,
            kv_bits=config.bits,
            sampling_profile=config.sampling_profile,
            model_id=config.model,
            adapter_path=config.adapter_path,
            revision=config.revision,
        )

    def list_models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self._runtime.settings.model_id,
                    "object": "model",
                    "owned_by": "embedded",
                }
            ],
        }

    def stream_response(
        self,
        *,
        model: str,  # ignored — embedded runtime uses its loaded model
        prompt: str,
        images: list[str],
        max_output_tokens: int,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        return self._runtime.stream_response_sync(
            prompt=prompt,
            images=images,
            max_output_tokens=max_output_tokens,
            system_prompt=system_prompt,
        )


def make_backend_client(
    settings: FrameworkSettings,
    *,
    in_process: bool = False,
) -> BackendClientProtocol:
    """Factory: returns an HTTP or embedded backend client based on flag."""
    if in_process:
        return EmbeddedBackendClient(settings)
    return BackendClient(settings)
