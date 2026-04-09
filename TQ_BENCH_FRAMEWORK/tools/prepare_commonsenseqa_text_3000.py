from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset


TARGET_TOTAL = 3000
SEED = 42


def _choices_block(options: list[str]) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    return "\n".join(f"{label}. {option}" for label, option in zip(labels, options))


def main() -> None:
    out_path = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "processed"
        / "commonsenseqa_text_3000"
        / "commonsenseqa_text_3000.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("tau/commonsense_qa", split="train")
    rows = [dict(row) for row in dataset]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    rows = rows[:TARGET_TOTAL]

    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            labels = [str(x) for x in row["choices"]["label"]]
            texts = [str(x) for x in row["choices"]["text"]]
            label_to_idx = {label: idx for idx, label in enumerate(labels)}
            answer_letter = chr(65 + label_to_idx[str(row["answerKey"])])
            record = {
                "sample_id": f"commonsenseqa_text_3000-{idx:04d}",
                "question": str(row["question"]),
                "answer": answer_letter,
                "metadata": {
                    "options": texts,
                    "choices_block": _choices_block(texts),
                    "question_concept": str(row.get("question_concept") or ""),
                    "source_dataset": "tau/commonsense_qa",
                    "source_split": "train",
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} samples to {out_path}")


if __name__ == "__main__":
    main()
