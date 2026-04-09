from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


TARGET_TOTAL = 1000
SEED = 42


def _index_to_letter(index: int) -> str:
    return chr(65 + index)


def main() -> None:
    out_path = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "processed"
        / "mmlu_text_1000"
        / "mmlu_text_1000.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("cais/mmlu", "all", split="test")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        grouped[str(row["subject"])].append(dict(row))

    rng = random.Random(SEED)
    subjects = sorted(grouped)
    base_quota = TARGET_TOTAL // len(subjects)
    remainder = TARGET_TOTAL % len(subjects)

    selected: list[dict] = []
    leftovers: list[dict] = []
    for idx, subject in enumerate(subjects):
        rows = grouped[subject][:]
        rng.shuffle(rows)
        take = base_quota + (1 if idx < remainder else 0)
        selected.extend(rows[:take])
        leftovers.extend(rows[take:])

    if len(selected) < TARGET_TOTAL:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: TARGET_TOTAL - len(selected)])

    selected = selected[:TARGET_TOTAL]
    selected.sort(key=lambda row: (str(row["subject"]), str(row["question"])))

    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(selected):
            choices = [str(choice) for choice in row["choices"]]
            answer_letter = _index_to_letter(int(row["answer"]))
            record = {
                "sample_id": f"mmlu_text_1000-{idx:04d}",
                "question": str(row["question"]),
                "answer": answer_letter,
                "metadata": {
                    "choice_A": choices[0],
                    "choice_B": choices[1],
                    "choice_C": choices[2],
                    "choice_D": choices[3],
                    "options": choices,
                    "subject": str(row["subject"]),
                    "source_dataset": "cais/mmlu",
                    "source_split": "test",
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} samples to {out_path}")


if __name__ == "__main__":
    main()
