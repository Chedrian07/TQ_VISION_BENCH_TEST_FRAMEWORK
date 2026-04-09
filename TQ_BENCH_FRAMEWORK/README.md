# TQ Bench Framework

Benchmark runner for comparing:

- baseline KV cache
- MLX native KV-cache quantization
- TurboQuant KV-cache quantization

against an OpenAI-compatible backend in `TQ_BACKEND/`.

## Core features

- benchmark selection by name
- `--num` / `--nums` per-benchmark sample cap
- runtime KV quantization reload through backend API
- dataset remote/local availability check
- dataset download and normalization into local JSONL
- sequential, streaming execution optimized for TTFT measurement
- structured `jsonl` logs and aggregate CSV/Markdown summaries
- resumable execution

## Unified dataset format

Each benchmark is expected to point to a local JSONL file. Each line should
look like:

```json
{
  "sample_id": "textvqa-0001",
  "question": "What word is on the sign?",
  "answer": "STOP",
  "images": ["images/0001.jpg"],
  "metadata": {
    "source_split": "validation"
  }
}
```

Supported answer fields:

- `"answer": "text"`
- `"answers": ["text1", "text2"]`

Supported image fields:

- `"image": "relative/or/absolute/path.jpg"`
- `"images": ["img1.jpg", "img2.jpg"]`

Relative image paths are resolved against the JSONL file directory.

Text-only benchmarks may omit `image` / `images` entirely.

## Example

```bash
cd /Users/kch3dri4n/project/TQ_VISION_BENCH_TEST_FRAMEWORK/TQ_BENCH_FRAMEWORK
uv sync
uv run tq-bench list-benchmarks
uv run tq-bench check-datasets --benchmarks textvqa,chartqa
uv run tq-bench prepare-datasets --benchmarks textvqa,chartqa --num 50

uv run tq-bench run \
  --benchmarks textvqa,chartqa \
  --num 50 \
  --sampling-profile controlled
```

## Dataset configuration

Each benchmark manifest points to an env var like `TEXTVQA_DATASET_FILE`.
Set it in `.env` or pass `--dataset-file textvqa=/abs/path/to/file.jsonl`.

`prepare-datasets` writes normalized JSONL files under:

- `datasets/processed/<benchmark>/<benchmark>.jsonl`

and updates `.env` automatically with the resolved file path.

Relevant timeout knobs:

- `TQ_BENCH_REQUEST_TIMEOUT_SECONDS` for inference requests
- `TQ_BENCH_RELOAD_TIMEOUT_SECONDS` for backend runtime reload / state checks
- `TQ_BENCH_CONNECT_TIMEOUT_SECONDS` for TCP connect timeout

### LongBench text subset

A deterministic 100-sample text-only benchmark derived from `THUDM/LongBench-v2`
can be prepared with:

```bash
uv run python tools/prepare_longbench_text_100.py
```

This writes:

- `datasets/processed/longbench_text_100/longbench_text_100.jsonl`

The prepared benchmark intentionally truncates each source context to keep
end-to-end KV-cache sweeps practical on local MLX hardware. It is meant for
relative quantization comparisons, not official LongBench reporting.

### MMLU text subset

A deterministic 1000-sample text-only multiple-choice benchmark can be prepared
from `cais/mmlu` with:

```bash
uv run python tools/prepare_mmlu_text_1000.py
```

This writes:

- `datasets/processed/mmlu_text_1000/mmlu_text_1000.jsonl`

The benchmark is intended as a larger-sample text-only KV-cache regression
probe with short prompts and no images.

### Mixed text-only suite

A mixed 3000-sample text-only suite can be prepared with:

```bash
uv run python tools/prepare_text_mixed_3000.py
```

This writes:

- `datasets/processed/text_mixed_3000/text_mixed_3000.jsonl`

The suite mixes:

- `cais/mmlu`
- `allenai/ai2_arc` (`ARC-Challenge`, `ARC-Easy`)
- `tau/commonsense_qa`
- `Rowan/hellaswag`

All tasks are normalized into a common multiple-choice format and scored with
the same option-match metric for relative KV-cache comparisons.

### 3000-sample single-benchmark text suites

The following larger text-only suites can also be prepared:

```bash
uv run python tools/prepare_mmlu_text_3000.py
uv run python tools/prepare_commonsenseqa_text_3000.py
uv run python tools/prepare_hellaswag_text_3000.py
```

They write:

- `datasets/processed/mmlu_text_3000/mmlu_text_3000.jsonl`
- `datasets/processed/commonsenseqa_text_3000/commonsenseqa_text_3000.jsonl`
- `datasets/processed/hellaswag_text_3000/hellaswag_text_3000.jsonl`

You can then point the manifest to it via:

- `LONGBENCH_TEXT_100_DATASET_FILE=/abs/path/to/longbench_text_100.jsonl`

or with a one-off override:

```bash
uv run tq-bench run \
  --benchmarks longbench_text_100 \
  --dataset-file longbench_text_100=/abs/path/to/longbench_text_100.jsonl
```

## Commands

### List manifests

```bash
uv run tq-bench list-benchmarks
```

### Check remote/local dataset status

```bash
uv run tq-bench check-datasets --benchmarks textvqa,chartqa,ai2d,docvqa
```

### Prepare normalized local datasets

```bash
uv run tq-bench prepare-datasets --benchmarks textvqa,chartqa --num 100 --overwrite
```

`prepare-datasets`는 expected row count를 검증하고, partial JSONL이나 stale image
artifacts를 재사용하지 않도록 안전하게 다시 생성한다.

### Run a benchmark sweep

```bash
uv run tq-bench run \
  --benchmarks textvqa,chartqa \
  --num 100 \
  --sampling-profile controlled
```

### Run only baseline

```bash
uv run tq-bench run \
  --benchmarks chartqa \
  --num 20 \
  --no-mlx \
  --no-turboquant
```

## Current dataset sources

Prepared automatically from public Hugging Face dataset repos:

- `textvqa` -> `lmms-lab/textvqa`
- `chartqa` -> `HuggingFaceM4/ChartQA`
- `ai2d` -> `lmms-lab/ai2d`
- `docvqa` -> `pixparse/docvqa-single-page-questions`
- `mathvista` -> `AI4Math/MathVista`
- `mmmu` -> `MMMU/MMMU`
- `ocrbench_v2` -> `morpheushoc/OCRBenchv2`
- `chartqapro` -> `ahmed-masry/ChartQAPro`

Notes:

- `textvqa`는 원본 `facebook/textvqa` 대신 public parquet mirror를 사용한다.
- `mmmu`는 multi-config dataset이라 prepare 시간이 상대적으로 길다.
- `HF_TOKEN`을 `.env`에 넣으면 rate limit과 다운로드 안정성이 더 좋아진다.

## Output layout

- prepared datasets:
  `datasets/processed/<benchmark>/<benchmark>.jsonl`
- raw run logs:
  `results/runs/<run_id>/raw/*.jsonl`
- aggregate CSV:
  `results/runs/<run_id>/aggregate/summary.csv`
- report markdown:
  `reports/<run_id>/summary.md`
- analysis markdown/json:
  `reports/<run_id>/analysis.md`, `reports/<run_id>/analysis.json`
- plot PNGs:
  `reports/<run_id>/*.png`

## Known limitations

- metric은 현재 lightweight local matcher 기준이다. official evaluator와 100% 일치하지 않을 수 있다.
- `TextVQA`는 official VQA soft accuracy가 아니라 exact-match 계열 내부 비교 점수를 사용한다.
- `OCRBench v2`도 official mixed metric이 아니라 내부 exact-match proxy를 사용한다.
- `DocVQA`의 `ANLS`는 내부 normalize 정책을 사용하므로 공식 리더보드 수치와 직접 비교하면 안 된다.
- percentage 값은 내부적으로 `45%`와 `0.45`를 모두 후보로 해석해 매칭한다.
- `MMMU`는 config를 모두 순회해 단일 JSONL로 합치므로 준비 시간이 길다.
- prepared datasets와 benchmark run artifacts는 로컬 산출물이며 Git에 포함되지 않는다.
