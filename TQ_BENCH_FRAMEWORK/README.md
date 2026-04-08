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

- metric은 현재 lightweight local matcher 기준이다. official evaluator와 100% 일치하도록 추가 보정이 필요할 수 있다.
- `MMMU`는 config를 모두 순회해 단일 JSONL로 합치므로 준비 시간이 길다.
- prepared datasets와 benchmark run artifacts는 로컬 산출물이며 Git에 포함되지 않는다.
