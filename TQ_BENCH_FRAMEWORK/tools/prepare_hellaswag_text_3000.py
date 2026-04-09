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
        / "hellaswag_text_3000"
        / "hellaswag_text_3000.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("Rowan/hellaswag", split="validation")
    rows = [dict(row) for row in dataset]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    rows = rows[:TARGET_TOTAL]

    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            options = [str(x) for x in row["endings"]]
            answer_letter = chr(65 + int(row["label"]))
            prompt = (
                "Complete the following passage with the most plausible ending.\n\n"
                f"Context:\n{row['ctx']}"
            )
            record = {
                "sample_id": f"hellaswag_text_3000-{idx:04d}",
                "question": prompt,
                "answer": answer_letter,
                "metadata": {
                    "options": options,
                    "choices_block": _choices_block(options),
                    "activity_label": str(row.get("activity_label") or ""),
                    "source_dataset": "Rowan/hellaswag",
                    "source_split": "validation",
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} samples to {out_path}")


if __name__ == "__main__":
    main()
