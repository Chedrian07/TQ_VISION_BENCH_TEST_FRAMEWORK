from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


SEED = 42
TARGETS = {
    "mmlu": 600,
    "arc_challenge": 600,
    "arc_easy": 600,
    "commonsenseqa": 600,
    "hellaswag": 600,
}


def _letter(index: int) -> str:
    return chr(65 + index)


def _choices_block(options: list[str]) -> str:
    return "\n".join(f"{_letter(i)}. {option}" for i, option in enumerate(options))


def _sample_mmlu(rng: random.Random, target: int) -> list[dict]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        grouped[str(row["subject"])].append(dict(row))

    subjects = sorted(grouped)
    per_subject = target // len(subjects)
    remainder = target % len(subjects)

    selected: list[dict] = []
    leftovers: list[dict] = []
    for idx, subject in enumerate(subjects):
        rows = grouped[subject][:]
        rng.shuffle(rows)
        take = per_subject + (1 if idx < remainder else 0)
        selected.extend(rows[:take])
        leftovers.extend(rows[take:])

    if len(selected) < target:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])

    records = []
    for i, row in enumerate(selected[:target]):
        options = [str(choice) for choice in row["choices"]]
        answer = _letter(int(row["answer"]))
        records.append(
            {
                "sample_id": f"mmlu-{i:04d}",
                "question": str(row["question"]),
                "answer": answer,
                "metadata": {
                    "options": options,
                    "choices_block": _choices_block(options),
                    "subject": str(row["subject"]),
                    "source_benchmark": "mmlu",
                    "source_dataset": "cais/mmlu",
                    "source_split": "test",
                },
            }
        )
    return records


def _sample_arc(rng: random.Random, config: str, prefix: str, target: int) -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", config, split="test")
    rows = [dict(row) for row in ds]
    rng.shuffle(rows)
    records = []
    for i, row in enumerate(rows[:target]):
        labels = [str(x) for x in row["choices"]["label"]]
        texts = [str(x) for x in row["choices"]["text"]]
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        answer = _letter(label_to_idx[str(row["answerKey"])])
        records.append(
            {
                "sample_id": f"{prefix}-{i:04d}",
                "question": str(row["question"]),
                "answer": answer,
                "metadata": {
                    "options": texts,
                    "choices_block": _choices_block(texts),
                    "source_benchmark": prefix,
                    "source_dataset": "allenai/ai2_arc",
                    "source_split": "test",
                },
            }
        )
    return records


def _sample_commonsenseqa(rng: random.Random, target: int) -> list[dict]:
    ds = load_dataset("tau/commonsense_qa", split="validation")
    rows = [dict(row) for row in ds]
    rng.shuffle(rows)
    records = []
    for i, row in enumerate(rows[:target]):
        labels = [str(x) for x in row["choices"]["label"]]
        texts = [str(x) for x in row["choices"]["text"]]
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        answer = _letter(label_to_idx[str(row["answerKey"])])
        records.append(
            {
                "sample_id": f"commonsenseqa-{i:04d}",
                "question": str(row["question"]),
                "answer": answer,
                "metadata": {
                    "options": texts,
                    "choices_block": _choices_block(texts),
                    "question_concept": str(row.get("question_concept") or ""),
                    "source_benchmark": "commonsenseqa",
                    "source_dataset": "tau/commonsense_qa",
                    "source_split": "validation",
                },
            }
        )
    return records


def _sample_hellaswag(rng: random.Random, target: int) -> list[dict]:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rows = [dict(row) for row in ds]
    rng.shuffle(rows)
    records = []
    for i, row in enumerate(rows[:target]):
        options = [str(x) for x in row["endings"]]
        answer = _letter(int(row["label"]))
        question = (
            "Complete the following passage with the most plausible ending.\n\n"
            f"Context:\n{row['ctx']}"
        )
        records.append(
            {
                "sample_id": f"hellaswag-{i:04d}",
                "question": question,
                "answer": answer,
                "metadata": {
                    "options": options,
                    "choices_block": _choices_block(options),
                    "activity_label": str(row.get("activity_label") or ""),
                    "source_benchmark": "hellaswag",
                    "source_dataset": "Rowan/hellaswag",
                    "source_split": "validation",
                },
            }
        )
    return records


def main() -> None:
    out_path = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "processed"
        / "text_mixed_3000"
        / "text_mixed_3000.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    records = []
    records.extend(_sample_mmlu(rng, TARGETS["mmlu"]))
    records.extend(_sample_arc(rng, "ARC-Challenge", "arc_challenge", TARGETS["arc_challenge"]))
    records.extend(_sample_arc(rng, "ARC-Easy", "arc_easy", TARGETS["arc_easy"]))
    records.extend(_sample_commonsenseqa(rng, TARGETS["commonsenseqa"]))
    records.extend(_sample_hellaswag(rng, TARGETS["hellaswag"]))

    records.sort(key=lambda row: row["sample_id"])

    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} samples to {out_path}")


if __name__ == "__main__":
    main()
