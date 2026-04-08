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
