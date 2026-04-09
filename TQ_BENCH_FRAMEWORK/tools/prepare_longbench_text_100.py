from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


MAX_CONTEXT_CHARS = 12_000

TARGET_COUNTS = {
    "Single-Document QA": 35,
    "Multi-Document QA": 25,
    "Long In-context Learning": 16,
    "Code Repository Understanding": 10,
    "Long-dialogue History Understanding": 8,
    "Long Structured Data Understanding": 6,
}


def main() -> None:
    out_path = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "processed"
        / "longbench_text_100"
        / "longbench_text_100.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        row = dict(row)
        source_context = str(row["context"])
        truncated_context = source_context[:MAX_CONTEXT_CHARS]
        row["_context"] = truncated_context
        row["_source_context_chars"] = len(source_context)
        row["_prompt_chars"] = sum(
            len(str(value))
            for value in (
                truncated_context,
                row["question"],
                row["choice_A"],
                row["choice_B"],
                row["choice_C"],
                row["choice_D"],
            )
        )
        grouped[str(row["domain"])].append(row)

    rng = random.Random(42)
    selected: list[dict] = []
    for domain, target_count in TARGET_COUNTS.items():
        rows = grouped[domain]
        if len(rows) < target_count:
            raise RuntimeError(
                f"Domain '{domain}' has only {len(rows)} rows, expected at least {target_count}."
            )
        rows = rows[:]
        rng.shuffle(rows)
        selected.extend(rows[:target_count])

    selected.sort(key=lambda row: str(row["_id"]))

    with out_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            record = {
                "sample_id": str(row["_id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "metadata": {
                    "context": str(row["_context"]),
                    "choice_A": str(row["choice_A"]),
                    "choice_B": str(row["choice_B"]),
                    "choice_C": str(row["choice_C"]),
                    "choice_D": str(row["choice_D"]),
                    "options": [
                        str(row["choice_A"]),
                        str(row["choice_B"]),
                        str(row["choice_C"]),
                        str(row["choice_D"]),
                    ],
                    "domain": str(row["domain"]),
                    "sub_domain": str(row["sub_domain"]),
                    "difficulty": str(row["difficulty"]),
                    "length": str(row["length"]),
                    "prompt_chars": int(row["_prompt_chars"]),
                    "source_context_chars": int(row["_source_context_chars"]),
                    "context_truncated": True,
                    "source_dataset": "THUDM/LongBench-v2",
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} samples to {out_path}")


if __name__ == "__main__":
    main()
