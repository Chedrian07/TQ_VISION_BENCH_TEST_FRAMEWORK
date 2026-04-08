# TQ_VISION_BENCH_TEST_FRAMEWORK

TurboQuant와 MLX native KV-cache quantization이 이미지 포함 VLM 추론 경로에 미치는 영향을 측정하기 위한 연구 저장소다.

현재 저장소는 네 영역으로 구성된다.

- `DOCUMENTS/`
  관련 논문 PDF
- `TQ_BACKEND/`
  `omlx.compat.vlm` 기반 OpenAI-compatible backend
- `TQ_BENCH_FRAMEWORK/`
  benchmark runner, dataset prepare/check, structured logging
- `RESEARCH_NOTE/`
  연구 계획, 실험 원칙, 아키텍처 문서

## 목표

이 저장소의 핵심 질문은 다음이다.

`TurboQuant`를 적용했을 때 이미지가 포함된 멀티모달 추론 경로에서 품질 저하 또는 시스템 성능 저하가 실제로 발생하는가?

이 질문에 답하기 위해 다음을 비교한다.

- baseline KV cache
- MLX native KV quantization
- TurboQuant KV quantization

그리고 다음을 동시에 측정한다.

- benchmark score
- `TTFT`
- total latency
- decode tokens/sec
- prompt/output token usage

## 현재 구현 상태

### Backend

`TQ_BACKEND`는 다음 기능을 제공한다.

- `POST /v1/responses`
- `GET /v1/models`
- `GET /v1/runtime`
- `POST /v1/runtime/reload`
- `GET /healthz`

중요한 정책:

- `Qwen3.5` thinking은 강제로 비활성
- runtime reload로 `model`, `kv quantization`, `sampling profile` 전환 가능
- 허용 KV bit:
  - `mlx`: `2`, `3`, `4`
  - `turboquant`: `2`, `2.5`, `3`, `3.5`, `4`

자세한 내용은 [TQ_BACKEND/README.md](/Users/kch3dri4n/project/TQ_VISION_BENCH_TEST_FRAMEWORK/TQ_BACKEND/README.md)를 본다.

### Benchmark Framework

`TQ_BENCH_FRAMEWORK`는 다음 기능을 제공한다.

- benchmark manifest registry
- `--num` 샘플 제한
- 특정 benchmark 선택 실행
- backend runtime reload를 이용한 baseline/MLX/TurboQuant sweep
- dataset remote/local availability check
- dataset download + unified JSONL prepare
- JSONL raw logs, CSV aggregate, Markdown summary 생성

자세한 사용법은 [TQ_BENCH_FRAMEWORK/README.md](/Users/kch3dri4n/project/TQ_VISION_BENCH_TEST_FRAMEWORK/TQ_BENCH_FRAMEWORK/README.md)를 본다.

## 빠른 시작

### 1. Backend 실행

```bash
cd /Users/kch3dri4n/project/TQ_VISION_BENCH_TEST_FRAMEWORK/TQ_BACKEND
uv sync
uv run serve.py
```

### 2. Dataset 점검

```bash
cd /Users/kch3dri4n/project/TQ_VISION_BENCH_TEST_FRAMEWORK/TQ_BENCH_FRAMEWORK
uv sync
uv run tq-bench check-datasets --benchmarks textvqa,chartqa,ai2d,docvqa
```

### 3. Dataset 준비

```bash
uv run tq-bench prepare-datasets --benchmarks textvqa,chartqa --num 50 --overwrite
```

### 4. 스모크 벤치 실행

```bash
uv run tq-bench run --benchmarks chartqa --num 5 --sampling-profile controlled
```

## 권장 벤치 순서

### 1차 코어

- `TextVQA`
- `ChartQA`
- `AI2D`
- `DocVQA`

### 2차 진단

- `OCRBench v2`
- `MathVista`
- `ChartQAPro` 조건부

### 3차 종합

- `MMMU`

## 결과 저장 위치

benchmark 결과는 기본적으로 아래에 저장된다.

- raw: `TQ_BENCH_FRAMEWORK/results/runs/<run_id>/raw/`
- summary csv: `TQ_BENCH_FRAMEWORK/results/runs/<run_id>/aggregate/summary.csv`
- summary markdown: `TQ_BENCH_FRAMEWORK/reports/<run_id>/summary.md`

## 참고 문서

- [RESEARCH_NOTE/TurboQuant_VLM_Benchmark_Research_Note.md](/Users/kch3dri4n/project/TQ_VISION_BENCH_TEST_FRAMEWORK/RESEARCH_NOTE/TurboQuant_VLM_Benchmark_Research_Note.md)
- [TQ_BACKEND/README.md](/Users/kch3dri4n/project/TQ_VISION_BENCH_TEST_FRAMEWORK/TQ_BACKEND/README.md)
- [TQ_BENCH_FRAMEWORK/README.md](/Users/kch3dri4n/project/TQ_VISION_BENCH_TEST_FRAMEWORK/TQ_BENCH_FRAMEWORK/README.md)
