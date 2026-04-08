"""
OpenAI Responses API server on top of `omlx.compat.vlm`.

This backend is benchmark-oriented:

1. One MLX VLM model is active per process.
2. Request concurrency is capped by configuration.
3. KV-cache quantization can be switched at runtime.
4. Qwen3.5 thinking is forced off for reproducible benchmarking.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional, Union

from engine._processor_shim import install_no_video_processor_shim
from engine.config import (
    ServerSettings,
    SettingsError,
    load_settings,
    replace_settings,
)

install_no_video_processor_shim()

import mlx.core as mx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from mlx_lm.sample_utils import make_presence_penalty
from pydantic import BaseModel, Field

from omlx.compat.vlm import (
    apply_chat_template,
    load,
    make_prompt_cache,
    quantize_prompt_cache,
    stream_generate,
)

try:
    SETTINGS = load_settings()
except SettingsError as exc:
    raise RuntimeError(f"Invalid backend configuration: {exc}") from exc

log = logging.getLogger("tq-serve")
logging.basicConfig(
    level=SETTINGS.log_level,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)


@dataclass(frozen=True)
class GenerationConfig:
    max_output_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float | None
    repetition_context_size: int
    presence_penalty: float | None
    presence_context_size: int
    prefill_step_size: int


class Runtime:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.model = None
        self.processor = None
        self.num_cache_layers: int | None = None
        self._load_lock = asyncio.Lock()
        self._generation_slots = asyncio.Semaphore(settings.max_concurrent_requests)
        self._generation_state = asyncio.Condition()
        self._active_generations = 0
        self._reloading = False

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def public_state(self) -> dict[str, Any]:
        sampling = self.settings.active_sampling
        return {
            "loaded": self.loaded,
            "model": {
                "id": self.settings.model_id,
                "adapter_path": self.settings.adapter_path,
                "revision": self.settings.revision,
            },
            "runtime": {
                "max_concurrent_requests": self.settings.max_concurrent_requests,
            },
            "sampling_profile": self.settings.sampling_profile,
            "sampling": {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "min_p": sampling.min_p,
                "presence_penalty": sampling.presence_penalty,
                "repetition_penalty": sampling.repetition_penalty,
                "max_output_tokens": self.settings.default_max_output_tokens,
            },
            "thinking": {
                "force_disable_thinking": self.settings.force_disable_thinking,
            },
            "kv_cache": {
                "scheme": self.settings.kv_quant_scheme,
                "omlx_scheme": self.settings.active_kv_quant_scheme,
                "mlx_vlm_scheme": self.settings.active_kv_quant_scheme,
                "bits": self.settings.kv_bits,
                "group_size": self.settings.kv_group_size,
                "quantized_kv_start": self.settings.quantized_kv_start,
                "turboquant_seed": self.settings.turboquant_seed,
                "turboquant_from_first_token": self.settings.turboquant_from_first_token,
            },
            "num_cache_layers": self.num_cache_layers,
            "env_files": self.settings.env_files,
        }

    async def ensure_loaded(self) -> None:
        if self.loaded:
            return

        async with self._load_lock:
            if self.loaded:
                return

            loop = asyncio.get_running_loop()
            model, processor, num_cache_layers = await loop.run_in_executor(
                None, self._load_sync
            )
            self.model = model
            self.processor = processor
            self.num_cache_layers = num_cache_layers

    def _load_sync(self):
        log.info(
            "Loading model %s (kv_scheme=%s kv_bits=%s profile=%s) ...",
            self.settings.model_id,
            self.settings.kv_quant_scheme,
            self.settings.kv_bits,
            self.settings.sampling_profile,
        )
        model, processor = load(
            self.settings.model_id,
            adapter_path=self.settings.adapter_path,
            revision=self.settings.revision,
            trust_remote_code=self.settings.trust_remote_code,
        )
        num_cache_layers = len(make_prompt_cache(model.language_model))
        log.info(
            "Model loaded. language_model has %d KV-cache slots.",
            num_cache_layers,
        )
        return model, processor, num_cache_layers

    def _unload_sync(self) -> None:
        self.model = None
        self.processor = None
        self.num_cache_layers = None
        gc.collect()
        mx.clear_cache()

    @asynccontextmanager
    async def generation_session(self):
        await self._generation_slots.acquire()
        try:
            async with self._generation_state:
                while self._reloading:
                    await self._generation_state.wait()
                self._active_generations += 1

            await self.ensure_loaded()
            yield
        finally:
            async with self._generation_state:
                self._active_generations -= 1
                if self._active_generations == 0:
                    self._generation_state.notify_all()
            self._generation_slots.release()

    async def reconfigure(self, new_settings: ServerSettings) -> tuple[dict[str, Any], dict[str, Any]]:
        async with self._generation_state:
            self._reloading = True
            while self._active_generations > 0:
                await self._generation_state.wait()

        try:
            async with self._load_lock:
                loop = asyncio.get_running_loop()
                previous_settings = self.settings
                previous_state = self.public_state()

                await loop.run_in_executor(None, self._unload_sync)
                self.settings = new_settings

                try:
                    model, processor, num_cache_layers = await loop.run_in_executor(
                        None, self._load_sync
                    )
                except Exception:
                    log.exception("runtime reload failed, attempting to restore previous model")
                    self.settings = previous_settings
                    try:
                        model, processor, num_cache_layers = await loop.run_in_executor(
                            None, self._load_sync
                        )
                        self.model = model
                        self.processor = processor
                        self.num_cache_layers = num_cache_layers
                    except Exception:
                        log.exception("failed to restore previous model after reload failure")
                    raise

                self.model = model
                self.processor = processor
                self.num_cache_layers = num_cache_layers
                self._generation_slots = asyncio.Semaphore(
                    self.settings.max_concurrent_requests
                )
                return previous_state, self.public_state()
        finally:
            async with self._generation_state:
                self._reloading = False
                self._generation_state.notify_all()

    def build_turbo_prompt_cache(self) -> list[Any]:
        if self.model is None or self.settings.kv_bits is None:
            raise RuntimeError("TurboQuant cache requested before runtime was initialized.")

        prompt_cache = make_prompt_cache(self.model.language_model)
        quantize_prompt_cache(
            prompt_cache,
            quantized_kv_start=0,
            kv_group_size=self.settings.kv_group_size,
            kv_bits=self.settings.kv_bits,
            kv_quant_scheme="turboquant",
        )
        return prompt_cache

    def build_generation_kwargs(self, params: GenerationConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_tokens": params.max_output_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "min_p": params.min_p,
            "prefill_step_size": params.prefill_step_size,
            "enable_thinking": not self.settings.force_disable_thinking,
        }

        repetition_penalty = params.repetition_penalty
        if repetition_penalty is not None and repetition_penalty not in (0.0, 1.0):
            kwargs["repetition_penalty"] = repetition_penalty
            kwargs["repetition_context_size"] = params.repetition_context_size

        logits_processors: list[Any] = []
        if params.presence_penalty is not None and params.presence_penalty != 0.0:
            logits_processors.append(
                make_presence_penalty(
                    params.presence_penalty,
                    params.presence_context_size,
                )
            )
        if logits_processors:
            kwargs["logits_processors"] = logits_processors

        if self.settings.use_turboquant_prompt_cache:
            kwargs["prompt_cache"] = self.build_turbo_prompt_cache()
        else:
            if self.settings.kv_bits is not None:
                kwargs["kv_bits"] = (
                    int(self.settings.kv_bits)
                    if self.settings.kv_quant_scheme == "mlx"
                    else self.settings.kv_bits
                )
                kwargs["kv_group_size"] = self.settings.kv_group_size
                kwargs["quantized_kv_start"] = self.settings.quantized_kv_start
            if self.settings.active_kv_quant_scheme is not None:
                kwargs["kv_quant_scheme"] = self.settings.active_kv_quant_scheme

        return kwargs


RUNTIME = Runtime(SETTINGS)


class InputTextPart(BaseModel):
    type: Literal["input_text", "text"]
    text: str


class _ImageURLObj(BaseModel):
    url: str


class InputImagePart(BaseModel):
    type: Literal["input_image", "image_url", "image"]
    image_url: Optional[Union[str, _ImageURLObj]] = None
    url: Optional[str] = None


InputPart = Union[InputTextPart, InputImagePart]


class InputMessage(BaseModel):
    role: Literal["user", "assistant", "system", "developer"] = "user"
    content: Union[str, list[InputPart]]


class ResponseCreateRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, list[InputMessage]]
    instructions: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = Field(default=None, ge=0.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: Optional[float] = Field(default=None, ge=0.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0)
    repetition_context_size: Optional[int] = Field(default=None, ge=1)
    presence_context_size: Optional[int] = Field(default=None, ge=1)
    max_output_tokens: Optional[int] = Field(default=None, ge=1, le=32_000)
    metadata: Optional[dict[str, Any]] = None


class RuntimeReloadRequest(BaseModel):
    model: Optional[str] = None
    adapter_path: Optional[str] = None
    revision: Optional[str] = None
    kv_quant_scheme: Optional[str] = None
    kv_bits: Optional[float] = None
    sampling_profile: Optional[
        Literal["controlled", "best_effort_general", "best_effort_reasoning"]
    ] = None


def _extract_image_source(part: InputImagePart) -> str | None:
    if isinstance(part.image_url, str):
        return part.image_url
    if isinstance(part.image_url, _ImageURLObj):
        return part.image_url.url
    return part.url


def _normalize_role(role: str) -> str:
    if role == "developer":
        return "system"
    return role


def build_prompt_and_images(
    req: ResponseCreateRequest,
    *,
    processor: Any,
    model_config: Any,
    enable_thinking: bool,
) -> tuple[str, list[str]]:
    messages: list[dict[str, Any]] = []

    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})

    if isinstance(req.input, str):
        messages.append({"role": "user", "content": req.input})
        images: list[str] = []
    else:
        images = []
        for msg in req.input:
            role = _normalize_role(msg.role)
            if isinstance(msg.content, str):
                messages.append({"role": role, "content": msg.content})
                continue

            parts_out: list[dict[str, Any]] = []
            for part in msg.content:
                if isinstance(part, InputTextPart):
                    parts_out.append({"type": "text", "text": part.text})
                elif isinstance(part, InputImagePart):
                    src = _extract_image_source(part)
                    if src:
                        images.append(src)
                        parts_out.append({"type": "image"})
            messages.append({"role": role, "content": parts_out})

    prompt = apply_chat_template(
        processor,
        model_config,
        messages,
        num_images=len(images),
        enable_thinking=enable_thinking,
    )
    return prompt, images


def resolve_generation_config(
    req: ResponseCreateRequest,
    settings: ServerSettings,
) -> GenerationConfig:
    defaults = settings.active_sampling
    top_p = defaults.top_p if req.top_p is None else req.top_p
    temperature = defaults.temperature if req.temperature is None else req.temperature
    if temperature > 0.0 and top_p == 0.0:
        raise HTTPException(
            status_code=400,
            detail="top_p must be > 0 when temperature > 0.",
        )

    return GenerationConfig(
        max_output_tokens=(
            settings.default_max_output_tokens
            if req.max_output_tokens is None
            else req.max_output_tokens
        ),
        temperature=temperature,
        top_p=top_p,
        top_k=defaults.top_k if req.top_k is None else req.top_k,
        min_p=defaults.min_p if req.min_p is None else req.min_p,
        presence_penalty=(
            defaults.presence_penalty
            if req.presence_penalty is None
            else req.presence_penalty
        ),
        repetition_penalty=(
            defaults.repetition_penalty
            if req.repetition_penalty is None
            else req.repetition_penalty
        ),
        repetition_context_size=(
            settings.default_repetition_context_size
            if req.repetition_context_size is None
            else req.repetition_context_size
        ),
        presence_context_size=(
            settings.default_presence_context_size
            if req.presence_context_size is None
            else req.presence_context_size
        ),
        prefill_step_size=settings.default_prefill_step_size,
    )


def authorize_request(request: Request) -> None:
    api_key = RUNTIME.settings.api_key
    if not api_key:
        return

    authorization = request.headers.get("authorization")
    if authorization == f"Bearer {api_key}":
        return

    raise HTTPException(status_code=401, detail="Invalid or missing API key.")


_SENTINEL = object()


async def run_generation(
    runtime: Runtime,
    prompt: str,
    images: list[str],
    params: GenerationConfig,
) -> AsyncIterator[Any]:
    await runtime.ensure_loaded()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
    model = runtime.model
    processor = runtime.processor
    if model is None or processor is None:
        raise RuntimeError("Runtime not initialized.")

    def worker() -> None:
        try:
            gen_kwargs = runtime.build_generation_kwargs(params)
            for result in stream_generate(
                model,
                processor,
                prompt,
                image=images if images else None,
                **gen_kwargs,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(result), loop).result()
        except Exception as exc:  # noqa: BLE001
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop).result()

    loop.run_in_executor(None, worker)

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _now() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _build_completed_response(
    *,
    response_id: str,
    model_id: str,
    created_at: int,
    text: str,
    input_tokens: int,
    output_tokens: int,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    msg_id = _new_id("msg")
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": model_id,
        "output": [
            {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "metadata": metadata or {},
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "temperature": None,
        "top_p": None,
        "truncation": "disabled",
    }


def _sse(event: str, data: dict[str, Any]) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


async def stream_sse(
    req: ResponseCreateRequest,
    prompt: str,
    images: list[str],
    params: GenerationConfig,
    *,
    response_id: str,
    created_at: int,
    model_id: str,
) -> AsyncIterator[bytes]:
    msg_id = _new_id("msg")
    seq = 0

    def nxt() -> int:
        nonlocal seq
        seq += 1
        return seq

    skeleton = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "model": model_id,
        "output": [],
        "metadata": req.metadata or {},
    }

    yield _sse(
        "response.created",
        {"type": "response.created", "sequence_number": nxt(), "response": skeleton},
    )
    yield _sse(
        "response.in_progress",
        {
            "type": "response.in_progress",
            "sequence_number": nxt(),
            "response": skeleton,
        },
    )
    yield _sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "sequence_number": nxt(),
            "output_index": 0,
            "item": {
                "type": "message",
                "id": msg_id,
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    )
    yield _sse(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "sequence_number": nxt(),
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )

    full_text_parts: list[str] = []
    last_result = None
    input_tokens = 0
    output_tokens = 0

    try:
        async for result in run_generation(RUNTIME, prompt, images, params):
            delta = result.text or ""
            last_result = result
            if delta:
                full_text_parts.append(delta)
                yield _sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": nxt(),
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta,
                    },
                )
    except Exception as exc:  # noqa: BLE001
        log.exception("generation failed")
        yield _sse(
            "response.failed",
            {
                "type": "response.failed",
                "sequence_number": nxt(),
                "response": {
                    **skeleton,
                    "status": "failed",
                    "error": {"code": "server_error", "message": str(exc)},
                },
            },
        )
        return

    full_text = "".join(full_text_parts)
    if last_result is not None:
        input_tokens = int(getattr(last_result, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(last_result, "generation_tokens", 0) or 0)

    yield _sse(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "sequence_number": nxt(),
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "text": full_text,
        },
    )
    yield _sse(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "sequence_number": nxt(),
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": full_text,
                "annotations": [],
            },
        },
    )
    yield _sse(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "sequence_number": nxt(),
            "output_index": 0,
            "item": {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": full_text,
                        "annotations": [],
                    }
                ],
            },
        },
    )

    completed = _build_completed_response(
        response_id=response_id,
        model_id=model_id,
        created_at=created_at,
        text=full_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        metadata=req.metadata,
    )
    yield _sse(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": nxt(),
            "response": completed,
        },
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    if RUNTIME.settings.preload_model:
        await RUNTIME.ensure_loaded()
    yield


app = FastAPI(title="TQ oMLX Responses API", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, **RUNTIME.public_state()}


@app.get("/v1/runtime")
async def get_runtime(request: Request) -> dict[str, Any]:
    authorize_request(request)
    return {"object": "runtime", **RUNTIME.public_state()}


@app.post("/v1/runtime/reload")
async def reload_runtime(request: Request, req: RuntimeReloadRequest) -> dict[str, Any]:
    authorize_request(request)

    try:
        new_settings = replace_settings(
            RUNTIME.settings,
            model_id=req.model,
            adapter_path=req.adapter_path,
            revision=req.revision,
            kv_quant_scheme=req.kv_quant_scheme,
            kv_bits=req.kv_bits,
            sampling_profile=req.sampling_profile,
        )
    except SettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    old_state, new_state = await RUNTIME.reconfigure(new_settings)
    return {
        "object": "runtime.reload",
        "previous": old_state,
        "current": new_state,
    }


@app.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    authorize_request(request)
    return {
        "object": "list",
        "data": [
            {
                "id": RUNTIME.settings.model_id,
                "object": "model",
                "created": _now(),
                "owned_by": "omlx",
            }
        ],
    }


@app.post("/v1/responses")
async def create_response(request: Request, req: ResponseCreateRequest):
    authorize_request(request)

    if req.stream:

        async def body() -> AsyncIterator[bytes]:
            async with RUNTIME.generation_session():
                settings = RUNTIME.settings
                if req.model and req.model != settings.model_id:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Unknown model '{req.model}'. This server serves '{settings.model_id}'.",
                    )

                params = resolve_generation_config(req, settings)
                model = RUNTIME.model
                processor = RUNTIME.processor
                if model is None or processor is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Model runtime failed to initialize.",
                    )

                prompt, images = build_prompt_and_images(
                    req,
                    processor=processor,
                    model_config=model.config,
                    enable_thinking=not settings.force_disable_thinking,
                )
                response_id = _new_id("resp")
                created_at = _now()

                async for chunk in stream_sse(
                    req,
                    prompt,
                    images,
                    params,
                    response_id=response_id,
                    created_at=created_at,
                    model_id=settings.model_id,
                ):
                    yield chunk

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    async with RUNTIME.generation_session():
        settings = RUNTIME.settings
        if req.model and req.model != settings.model_id:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown model '{req.model}'. This server serves '{settings.model_id}'.",
            )

        params = resolve_generation_config(req, settings)
        model = RUNTIME.model
        processor = RUNTIME.processor
        if model is None or processor is None:
            raise HTTPException(status_code=500, detail="Model runtime failed to initialize.")

        try:
            prompt, images = build_prompt_and_images(
                req,
                processor=processor,
                model_config=model.config,
                enable_thinking=not settings.force_disable_thinking,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Bad input: {exc}") from exc

        full_text_parts: list[str] = []
        last_result = None

        try:
            async for result in run_generation(RUNTIME, prompt, images, params):
                if result.text:
                    full_text_parts.append(result.text)
                last_result = result
        except Exception as exc:  # noqa: BLE001
            log.exception("generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        full_text = "".join(full_text_parts)
        input_tokens = int(getattr(last_result, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(last_result, "generation_tokens", 0) or 0)

        payload = _build_completed_response(
            response_id=_new_id("resp"),
            model_id=settings.model_id,
            created_at=_now(),
            text=full_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=req.metadata,
        )
        return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host=RUNTIME.settings.host,
        port=RUNTIME.settings.port,
        reload=False,
    )
