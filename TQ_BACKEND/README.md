# TQ Backend

`TQ_BACKEND`는 `mlx_vlm` 기반의 OpenAI-compatible inference backend다.

이 backend는 단순 서빙 서버가 아니라 benchmark control plane 역할을 한다.

## 제공 API

- `GET /healthz`
- `GET /v1/models`
- `GET /v1/runtime`
- `POST /v1/runtime/reload`
- `POST /v1/responses`

## 핵심 정책

### Qwen3.5 non-thinking 강제

이 backend는 benchmark 재현성을 위해 `Qwen3.5` thinking mode를 항상 비활성화한다.

관련 설정:

- `TQ_FORCE_DISABLE_THINKING=true`

### 허용 KV quantization 설정

- `none`
- `mlx` with bits `2`, `3`, `4`
- `turboquant` with bits `2`, `2.5`, `3`, `3.5`, `4`

지원되지 않는 값은 `POST /v1/runtime/reload`에서 `400`으로 거부된다.

현재 backend 기본 정책:

- `TurboQuant`는 첫 토큰부터 KV quantization 적용
- `MLX native KV quant`도 `TQ_QUANTIZED_KV_START=0`으로 첫 토큰부터 적용

### sampling profile

현재 지원:

- `controlled`
- `best_effort_general`
- `best_effort_reasoning`

profile별 기본값은 `.env`와 `.env.example`에 정의돼 있다.

## 환경 설정

기본 설정 파일:

- [TQ_BACKEND/.env.example](/Users/kch3dri4n/project/TQ_VISION_BENCH_TEST_FRAMEWORK/TQ_BACKEND/.env.example)
- [TQ_BACKEND/.env](/Users/kch3dri4n/project/TQ_VISION_BENCH_TEST_FRAMEWORK/TQ_BACKEND/.env)

중요 변수:

- `TQ_MODEL`
- `TQ_KV_QUANT_SCHEME`
- `TQ_KV_BITS`
- `TQ_SAMPLING_PROFILE`
- `TQ_FORCE_DISABLE_THINKING`
- `TQ_API_KEY`

## 실행

```bash
cd /Users/kch3dri4n/project/TQ_VISION_BENCH_TEST_FRAMEWORK/TQ_BACKEND
uv sync
uv run serve.py
```

또는:

```bash
uv run uvicorn serve:app --host 0.0.0.0 --port 8000
```

## runtime reload 예시

MLX native 4bit:

```bash
curl http://localhost:8000/v1/runtime/reload \
  -H "Authorization: Bearer api" \
  -H "Content-Type: application/json" \
  -d '{
    "kv_quant_scheme": "mlx",
    "kv_bits": 4,
    "sampling_profile": "controlled"
  }'
```

TurboQuant 2.5bit:

```bash
curl http://localhost:8000/v1/runtime/reload \
  -H "Authorization: Bearer api" \
  -H "Content-Type: application/json" \
  -d '{
    "kv_quant_scheme": "turboquant",
    "kv_bits": 2.5,
    "sampling_profile": "best_effort_general"
  }'
```

baseline:

```bash
curl http://localhost:8000/v1/runtime/reload \
  -H "Authorization: Bearer api" \
  -H "Content-Type: application/json" \
  -d '{
    "kv_quant_scheme": "none",
    "sampling_profile": "controlled"
  }'
```

## health/runtime 확인

```bash
curl http://localhost:8000/healthz

curl http://localhost:8000/v1/runtime \
  -H "Authorization: Bearer api"
```

## 주의사항

- generation은 직렬화되어 있다.
- runtime reload도 generation과 직렬화된다.
- benchmark 목적상 thinking은 강제로 비활성화된다.
- local `.env`는 Git에 포함되지 않는다. 공유가 필요한 값만 `.env.example`에 반영해야 한다.
