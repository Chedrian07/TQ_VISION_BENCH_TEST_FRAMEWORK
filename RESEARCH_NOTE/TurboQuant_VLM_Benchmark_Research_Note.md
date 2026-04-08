# TurboQuant VLM Benchmark Research Note

## 1. 문서 목적

이 문서는 `TurboQuant`가 이미지가 포함된 멀티모달 추론 경로에서 실제 성능 저하를 유발하는지 검증하고, 저하가 확인되면 재현 가능한 방식으로 루트코즈를 좁혀가기 위한 연구 계획서이자 인수인계 문서다.

이 문서의 목표는 다음과 같다.

- 현재 연구 질문과 가설을 명확히 정리한다.
- 현재 워크스페이스에서 이미 구현된 백엔드 기능과 제약을 기록한다.
- 향후 구현할 벤치마크 프레임워크의 아키텍처를 설계 수준에서 명확히 남긴다.
- 다른 연구자가 이 저장소를 이어받아도 같은 기준으로 실험을 계속할 수 있도록 재현 규칙을 문서화한다.

## 2. 현재 연구 질문

핵심 질문은 다음 한 문장으로 요약된다.

`TurboQuant`를 적용했을 때, 텍스트 전용 경로가 아니라 이미지가 포함된 VLM 추론 경로에서 품질 또는 시스템 성능이 유의미하게 악화되는가?

이 질문은 다시 네 개의 하위 질문으로 쪼갠다.

1. `TurboQuant`가 VLM 품질을 실제로 떨어뜨리는가?
2. 떨어진다면 어떤 시각적 하위 능력에서 먼저 드러나는가?
3. 품질 저하와 시스템 성능 저하는 같은 원인인가, 별도 원인인가?
4. 저하가 `TurboQuant` 특이 현상인가, 아니면 저비트 KV quantization 전반의 일반적 현상인가?

## 3. 연구 가설

이 연구는 아직 `TurboQuant`의 "이미지 경로 열화"가 알려진 버그라는 전제를 두지 않는다. 이 문제는 현재 시점에서 검증 대상 가설이다.

우리가 검증하려는 주요 가설은 다음과 같다.

### H1. OCR 계열 민감도 가설

작고 조밀한 텍스트를 많이 포함하는 입력에서는 저비트 KV cache가 더 쉽게 성능 저하를 유발할 수 있다.

예상 징후:

- `TextVQA`, `DocVQA`, `OCRBench v2`가 먼저 무너진다.
- `AI2D`는 상대적으로 덜 흔들린다.

### H2. 정밀 시각 정보 가설

차트 축, 범례, 미세 숫자, 세부 영역 참조가 중요한 과제에서는 quantization noise가 더 크게 드러날 수 있다.

예상 징후:

- `ChartQA`에서 baseline 대비 하락폭이 커진다.
- 필요 시 `ChartQAPro`에서 하락이 더 확대된다.

### H3. 구조적 추론 분리 가설

OCR 문제와 별개로 구조적 다이어그램 이해 자체가 무너지는지 확인해야 한다.

예상 징후:

- `AI2D`가 크게 하락하면 단순 OCR 문제가 아니라 시각-구조 추론 경로 자체 문제일 가능성이 있다.

### H4. 시스템 병목 가설

품질과 별개로 이미지 토큰이 많은 프롬프트에서는 `prefill`, `TTFT`, `decode tok/s`, 메모리 사용량의 trade-off가 텍스트-only보다 다르게 나타날 수 있다.

예상 징후:

- 품질은 유지되지만 `TTFT`나 `prefill latency`만 증가한다.
- `TurboQuant`와 `MLX native KV quant`가 같은 bit에서도 다른 latency profile을 보인다.

## 4. 현재 워크스페이스 상태

현재 저장소는 다음 세 영역으로 구성된다.

- `DOCUMENTS/`
- `TQ_BACKEND/`
- `TQ_BENCH_FRAMEWORK/`
- `RESEARCH_NOTE/`

### 4.1 DOCUMENTS

관련 논문 PDF가 들어 있다.

- `TurboQuan.pdf`
- `QJL.pdf`
- `PolarQuant.pdf`

이 문헌들은 주로 KV cache quantization의 이론과 텍스트 long-context 실험을 다루며, 이미지 경로 성능 저하를 직접 다루지는 않는다. 따라서 "멀티모달 이미지 경로에서 실제로 어떤 현상이 일어나는가"는 본 저장소에서 직접 검증해야 하는 연구 공백이다.

### 4.2 TQ_BACKEND

현재 백엔드는 `mlx_vlm` 기반의 OpenAI-compatible Responses API 서버다.

핵심 특징:

- `POST /v1/responses` 제공
- `GET /v1/models` 제공
- `GET /v1/runtime` 제공
- `POST /v1/runtime/reload` 제공
- `GET /healthz` 제공

현재 백엔드는 다음 정책을 갖는다.

- 모델은 프로세스당 1개 활성화
- generation은 직렬화
- runtime reload를 통해 모델/quantization/sampling profile 교체 가능
- `Qwen3.5`는 항상 non-thinking mode 강제
- `TurboQuant`와 `MLX native KV quant` 모두 서버 설정으로 전환 가능

### 4.3 TQ_BENCH_FRAMEWORK

아직 실질적인 벤치 실행 코드는 없다.

현재 상태:

- `.env`
- `.env.example`
- `.python-version`
- `pyproject.toml`

즉, 벤치 프레임워크는 이제부터 설계 및 구현해야 하는 단계다.

## 5. 현재 백엔드의 실험적 의미

현재 백엔드는 단순 서빙 서버가 아니라, "실험 컨트롤 플레인"으로 간주해야 한다.

왜냐하면 이 서버는 단순히 질의응답만 하는 것이 아니라 다음 요소를 실험 중에 바꿀 수 있기 때문이다.

- 모델 ID
- adapter path
- revision
- KV quantization scheme
- KV bits
- sampling profile

즉, 벤치 프레임워크는 모델 추론 로직을 직접 갖지 않고, 이 백엔드를 "원격 실행기"처럼 사용하면 된다.

## 6. 현재 백엔드에서 허용되는 KV-cache 설정

현재 백엔드 정책은 다음과 같이 고정한다.

### 6.1 MLX native KV quantization

허용 bit:

- `2`
- `3`
- `4`

의미:

- `mlx` 또는 `uniform` 계열의 기존 네이티브 KV quantization 경로
- fractional bit는 허용하지 않음

### 6.2 TurboQuant

허용 bit:

- `2`
- `2.5`
- `3`
- `3.5`
- `4`

의미:

- fractional bit 실험 가능
- `TurboQuant` 전용 low-bit 비교군 구성 가능

### 6.3 Baseline

baseline은 다음으로 정의한다.

- `scheme = none`
- `kv_bits = null`

즉, quantization을 적용하지 않은 full-precision KV cache 기준선이다.

## 7. Qwen3.5 관련 정책

현재 연구에서는 `Qwen3.5` reasoning 기능을 항상 끈다.

이유:

- thinking mode는 생성 길이, token budget, 샘플링 경로, latency에 추가 분산을 만든다.
- 현재 연구의 1차 목적은 `KV quantization`의 영향 분리이지 reasoning mode 비교가 아니다.
- 재현성과 비교 가능성을 위해 non-thinking mode를 고정하는 편이 낫다.

현재 backend 정책:

- `TQ_FORCE_DISABLE_THINKING=true`
- prompt template에도 `enable_thinking=false`가 반영됨

주의:

`best_effort_reasoning` 프로파일은 reasoning task에 권장된 샘플링 값을 쓰되, 여전히 thinking 자체는 비활성 상태다.

즉 이 프로파일은 "reasoning benchmark용 non-thinking sampling profile"로 이해해야 한다.

## 8. 샘플링 프로파일 정책

현재 백엔드는 세 가지 샘플링 프로파일을 갖는다.

### 8.1 controlled

목적:

- 주 실험
- root cause 분석
- 정량 비교의 기준선

기본값:

- `temperature = 0.0`
- `top_p = 1.0`
- `top_k = 0`
- `min_p = 0.0`
- `presence_penalty = disabled`
- `repetition_penalty = disabled`

### 8.2 best_effort_general

목적:

- 자연 이미지 기반 일반 태스크 보조 실험
- 실제 모델 권장값 기준 확인

기본값:

- `temperature = 0.7`
- `top_p = 0.8`
- `top_k = 20`
- `min_p = 0.0`
- `presence_penalty = 1.5`
- `repetition_penalty = disabled`

### 8.3 best_effort_reasoning

목적:

- 시각 수학/종합 reasoning 계열 보조 실험

기본값:

- `temperature = 1.0`
- `top_p = 0.95`
- `top_k = 20`
- `min_p = 0.0`
- `presence_penalty = 1.5`
- `repetition_penalty = disabled`

## 9. 권장 벤치마크 구성

### 9.1 최종 권장 구조

#### 1차 코어

- `TextVQA`
- `ChartQA`
- `AI2D`
- `DocVQA`

이 단계의 목적은 "어디서 처음 무너지는지"를 빠르게 찾는 것이다.

#### 2차 진단

- `OCRBench v2`
- `MathVista`
- `ChartQAPro` 조건부

이 단계의 목적은 1차에서 드러난 문제의 성격을 좁히는 것이다.

규칙:

- OCR 계열 하락이 보이면 `OCRBench v2`
- 차트 계열 하락이 보이면 `ChartQAPro`
- reasoning 계열 분리 확인이 필요하면 `MathVista`

#### 3차 종합

- `MMMU`

이 단계의 목적은 "최종적으로 종합 멀티도메인 성능이 유지되는가"를 확인하는 것이다.

### 9.2 왜 이 조합을 선택하는가

#### TextVQA

- 자연 이미지 + 작은 텍스트
- OCR sensitivity 확인에 좋다.

#### DocVQA

- 문서 레이아웃 + dense text
- 문서형 prefill 부담과 OCR/레이아웃 민감도를 본다.

#### ChartQA

- 숫자, 범례, 축, 도형적 참조가 섞인다.
- 정밀 시각 정보 손실이 잘 드러난다.

#### AI2D

- 다이어그램 구조 이해
- OCR 문제가 아니라 구조적 시각 추론이 깨지는지 분리하는 역할을 한다.

#### OCRBench v2

- 대표 점수용보다는 진단용
- OCR 계열 하락의 원인이 진짜 OCR인지 확인한다.

#### MathVista

- 시각적 reasoning 강도가 높은 과제
- OCR과 다른 축의 실패를 분리한다.

#### MMMU

- 원인 분석보다는 최종 종합 검증용
- 넓지만 해석력이 낮아서 1차보다는 3차에 적합하다.

#### ChartQAPro

- `ChartQA`에서 문제 신호가 생겼을 때 후속 확대 실험용
- 초기부터 항상 돌리기에는 비용 대비 효율이 낮다.

## 10. 우리가 비교할 실험 축

최소 비교 매트릭스는 다음과 같다.

### 10.1 Quantization 축

- `baseline`
- `mlx-2`
- `mlx-3`
- `mlx-4`
- `tq-2`
- `tq-2.5`
- `tq-3`
- `tq-3.5`
- `tq-4`

### 10.2 Sampling 축

- `controlled` 필수
- `best_effort_general` 선택
- `best_effort_reasoning` 선택

기본 원칙:

- 결론은 항상 `controlled` 결과를 우선한다.
- best-effort는 보조 분석과 현실적 사용성 확인용이다.

### 10.3 Task 축

- OCR-heavy
- document
- chart
- diagram
- visual reasoning
- broad multimodal comprehensive

## 11. 측정해야 할 지표

이 연구는 단순 accuracy 비교로 끝내면 안 된다. 품질과 시스템 성능을 분리 측정해야 한다.

### 11.1 품질 지표

각 벤치에 맞는 공식 metric을 사용한다.

- `TextVQA`: exact match 또는 official normalization
- `DocVQA`: `ANLS`
- `ChartQA`: numeric tolerance 포함한 official metric
- `AI2D`: exact match
- `MathVista`: official matcher
- `MMMU`: official matcher
- `OCRBench v2`: benchmark가 정의하는 OCR score
- `ChartQAPro`: benchmark 정의 metric

### 11.2 시스템 지표

반드시 수집해야 할 시스템 지표는 다음과 같다.

- `TTFT`
- `end-to-end latency`
- `decode tok/s`
- `prompt tokens`
- `output tokens`
- `image count`
- `image resolution bucket`
- `kv quant scheme`
- `kv bits`
- `sampling profile`
- `hardware metadata`

가능하면 추가로 수집할 지표:

- `prefill latency`
- `generation latency`
- `peak memory`

주의:

현재 backend의 공개 API payload는 `prompt_tokens`와 `generation_tokens`는 간접 확보 가능하지만, `peak_memory`, `prompt_tps`, `generation_tps`는 외부로 바로 내보내지 않는다. 이 부분은 후속 backend 확장 또는 runner의 client-side timing으로 보완해야 한다.

## 12. 권장 측정 방식

### 12.1 기본 방침

벤치 프레임워크는 가능하면 `stream=true` 경로를 사용한다.

이유:

- `TTFT`를 구하려면 첫 token 도착 시점을 알아야 한다.
- non-stream 방식은 최종 응답만 보이므로 prefill/decode 구분이 약해진다.

### 12.2 권장 타임라인

샘플 단위로 다음 시간을 기록한다.

- `t_request_start`
- `t_first_delta`
- `t_response_done`

이로부터 계산한다.

- `TTFT = t_first_delta - t_request_start`
- `total_latency = t_response_done - t_request_start`
- `decode_latency = t_response_done - t_first_delta`
- `decode_tok_per_sec = output_tokens / decode_latency`

### 12.3 주의할 점

- single-turn 요청만 사용
- prompt cache 재사용 금지
- vision feature cache 재사용 금지
- 같은 샘플을 비교할 때 prompt template를 quantization 설정 간 동일하게 유지

## 13. 권장 벤치 프레임워크 아키텍처

아키텍처는 5계층으로 나누는 것이 가장 관리하기 쉽다.

### 13.1 Layer 1: Manifest Layer

역할:

- 벤치 정의를 선언적으로 관리

포함 내용:

- 데이터셋 이름
- split
- subset 규칙
- metric 종류
- prompt template
- sampling profile 기본값
- output parser

예상 파일 예시:

- `configs/benchmarks/textvqa.yaml`
- `configs/benchmarks/chartqa.yaml`
- `configs/benchmarks/ai2d.yaml`

### 13.2 Layer 2: Dataset Adapter Layer

역할:

- 서로 다른 데이터셋을 공통 샘플 포맷으로 변환

공통 sample schema 예시:

- `sample_id`
- `benchmark`
- `question`
- `image_paths`
- `ground_truth`
- `metadata`

이 레이어의 목적은 모델 호출 로직이 데이터셋 구조를 몰라도 되게 만드는 것이다.

### 13.3 Layer 3: Runtime Controller Layer

역할:

- 백엔드 설정을 실험 셀마다 명시적으로 바꾸는 컨트롤 플레인

사용 API:

- `GET /v1/runtime`
- `POST /v1/runtime/reload`

실험 셀 시작 전 절차:

1. 원하는 `model`, `kv scheme`, `bits`, `sampling profile`로 reload
2. 응답이 성공했는지 확인
3. 기록 파일에 runtime snapshot 저장

### 13.4 Layer 4: Runner Layer

역할:

- 샘플을 순회하며 backend API 호출
- stream 수집
- raw output 저장

runner가 담당할 일:

- request 생성
- first-token timing 기록
- 최종 text 복원
- raw response 저장
- 실패 재시도 정책 적용

### 13.5 Layer 5: Metric and Reporting Layer

역할:

- 데이터셋별 score 계산
- 시스템 지표 집계
- delta 분석

산출물:

- `raw jsonl`
- `aggregate csv/parquet`
- `markdown summary`
- `plots`

## 14. 권장 디렉터리 구조

아직 구현 전이지만 다음 구조를 권장한다.

```text
TQ_BENCH_FRAMEWORK/
├── configs/
│   ├── benchmarks/
│   ├── experiments/
│   └── profiles/
├── datasets/
│   ├── loaders/
│   ├── adapters/
│   └── cache/
├── prompts/
├── runner/
├── metrics/
├── analysis/
├── reports/
├── results/
│   ├── raw/
│   ├── aggregate/
│   └── plots/
└── scripts/
```

## 15. 실험 실행 전략

### 15.1 Stage A: 빠른 스크리닝

각 코어 벤치에서 `200~300`개 정도의 stratified subset을 사용한다.

목적:

- 어떤 축에서 문제 신호가 나오는지 빠르게 찾기
- 모든 조합을 full run 하지 않기

### 15.2 Stage B: 진단 확대

Stage A에서 하락이 확인된 벤치만 확대한다.

예:

- `TextVQA`, `DocVQA` 하락 -> `OCRBench v2`
- `ChartQA` 하락 -> `ChartQAPro`
- `MathVista` 하락 -> reasoning profile 재검토

### 15.3 Stage C: 최종 종합 평가

살아남은 설정만 `MMMU`에 투입한다.

목적:

- 최종 종합 성능 유지 여부 확인
- 논문/보고서용 대표 비교표 구성

## 16. 결과 해석 규칙

실험 결과는 다음 규칙으로 해석한다.

### 16.1 OCR만 하락

가능 원인:

- 작은 시각 텍스트 보존 실패
- dense visual token 민감도 증가

후속 액션:

- `OCRBench v2`
- 해상도 bucket별 분석
- 짧은 답변 vs 긴 답변 분리

### 16.2 Chart 계열만 하락

가능 원인:

- 숫자/축/범례/정밀 참조 손실

후속 액션:

- `ChartQAPro`
- number extraction error analysis

### 16.3 AI2D까지 함께 하락

가능 원인:

- 구조적 시각 추론 경로 자체 영향
- 단순 OCR 문제가 아님

후속 액션:

- diagram-specific qualitative audit
- multi-hop visual reasoning error sampling

### 16.4 품질은 유지되지만 latency만 악화

가능 원인:

- prefill 병목
- TurboQuant codec/kernel overhead
- image token 수 증가에 따른 비용 확대

후속 액션:

- `TTFT`, `decode tok/s`, prompt length correlation 분석

### 16.5 MLX와 TurboQuant가 둘 다 비슷하게 하락

가능 원인:

- 일반적인 저비트 quantization 한계

### 16.6 TurboQuant만 유독 더 나쁨

가능 원인:

- TurboQuant-specific codec/path issue
- attention layer별 민감도 차이

## 17. 재현성 규칙

후속 연구자도 반드시 지켜야 할 재현 규칙은 다음과 같다.

- 동일 하드웨어 조건 기록
- macOS 버전 기록
- Apple Silicon 기종과 메모리 크기 기록
- `mlx`, `mlx_vlm`, backend commit hash 기록
- 모든 실험에 runtime snapshot 저장
- subset seed 고정
- single-turn 요청만 사용
- prompt template 변경 금지
- 같은 비교군에서는 sampling profile 고정

## 18. 로그와 저장 포맷 권장안

### 18.1 raw result

샘플 단위 `jsonl`

필드 예시:

- `run_id`
- `benchmark`
- `sample_id`
- `model_id`
- `kv_scheme`
- `kv_bits`
- `sampling_profile`
- `question`
- `prediction`
- `ground_truth`
- `ttft_ms`
- `total_latency_ms`
- `decode_tps`
- `prompt_tokens`
- `output_tokens`
- `error`

### 18.2 aggregate result

벤치 단위 `csv` 또는 `parquet`

필드 예시:

- `benchmark`
- `model_id`
- `kv_scheme`
- `kv_bits`
- `sampling_profile`
- `score`
- `score_delta_vs_baseline`
- `avg_ttft_ms`
- `avg_total_latency_ms`
- `avg_decode_tps`
- `num_samples`

## 19. 현재까지의 의사결정 요약

현재까지 합의된 내용은 다음과 같다.

- backend는 `mlx_vlm` 기반 OpenAI Responses API로 통일
- backend가 quantization/runtime switching control plane 역할을 담당
- Qwen3.5는 non-thinking mode 강제
- 실험의 주 결과는 `controlled` 프로파일 기준으로 비교
- benchmark sequence는 `1차 코어 -> 2차 진단 -> 3차 종합`
- `MMMU`는 종합 검증용으로만 사용
- `ChartQAPro`는 조건부 확장

## 20. 향후 구현 우선순위

### Priority 0

- backend 안정성 유지
- runtime reload와 OpenAI-compatible inference 경로 유지

### Priority 1

- `TQ_BENCH_FRAMEWORK` skeleton 구현
- manifest loader
- runtime controller
- stream runner

### Priority 2

- `TextVQA`, `ChartQA`, `AI2D`, `DocVQA` adapter 구현
- official metric wrapper 구현

### Priority 3

- subset screening pipeline 구현
- aggregate reporting 구현

### Priority 4

- `MathVista`, `MMMU`, `OCRBench v2`, `ChartQAPro` 확장
- qualitative analysis report 자동 생성

## 21. 아직 남아 있는 열린 문제

아직 결정되지 않았거나 추가 검증이 필요한 문제는 다음과 같다.

- `DocVQA` 데이터 접근 정책과 자동 다운로드 여부
- benchmark별 official metric 구현 방식
- 결과 저장 포맷을 `csv` 중심으로 갈지 `parquet`까지 포함할지
- backend가 `peak_memory`, `prompt_tps`, `generation_tps`를 API로 직접 노출할지 여부
- multi-image benchmark를 초기에 포함할지 여부

## 22. 다음 작업 권장 순서

실제 구현은 다음 순서가 가장 안전하다.

1. `TQ_BENCH_FRAMEWORK` 디렉터리 구조 생성
2. benchmark manifest 포맷 확정
3. backend runtime controller client 구현
4. stream-based unified runner 구현
5. core 4개 벤치 adapter 구현
6. subset screening 실행
7. 진단 벤치 확장
8. MMMU 종합 검증

## 23. 결론

이 연구의 핵심은 "TurboQuant가 이미지 경로에서 진짜 문제를 만드는가"를 묻는 것이다. 이 질문은 정확도 하나로 답할 수 없고, 품질 저하와 시스템 성능 저하를 분리하여 봐야 한다.

현재 저장소는 그 연구를 시작하기 위한 기반을 이미 일부 갖췄다.

- backend는 준비되었다.
- quantization switching control plane도 있다.
- sampling profile과 non-thinking 정책도 정리되었다.

이제 남은 일은 benchmark framework를 체계적으로 구현하고, 코어 벤치부터 순서대로 신호를 수집하는 것이다.

이 문서는 그 출발 기준점이며, 이후 연구자는 반드시 이 문서의 실험 통제 원칙과 해석 규칙을 유지한 채 확장 작업을 진행해야 한다.
