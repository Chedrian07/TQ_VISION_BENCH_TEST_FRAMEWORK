from __future__ import annotations

import math
import re


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s.%/-]+", re.UNICODE)
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, a_char in enumerate(a, start=1):
        current = [i]
        for j, b_char in enumerate(b, start=1):
            cost = 0 if a_char == b_char else 1
            current.append(
                min(
                    prev[j] + 1,
                    current[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = current
    return prev[-1]


def score_exact_match(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    normalized_answers = {normalize_text(answer) for answer in answers}
    return 1.0 if normalized_prediction in normalized_answers else 0.0


def score_anls(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    best = 0.0
    for answer in answers:
        normalized_answer = normalize_text(answer)
        if not normalized_answer and not normalized_prediction:
            best = max(best, 1.0)
            continue
        distance = levenshtein_distance(normalized_prediction, normalized_answer)
        denominator = max(len(normalized_prediction), len(normalized_answer), 1)
        score = 1.0 - (distance / denominator)
        if score < 0.5:
            score = 0.0
        best = max(best, score)
    return best


def _parse_number(text: str) -> float | None:
    match = _NUMBER_RE.search(normalize_text(text))
    if not match:
        return None
    token = match.group(0).replace(",", "")
    if token.endswith("%"):
        return float(token[:-1]) / 100.0
    return float(token)


def score_numeric_relaxed_accuracy(prediction: str, answers: list[str]) -> float:
    prediction_number = _parse_number(prediction)
    if prediction_number is None:
        return score_exact_match(prediction, answers)

    for answer in answers:
        answer_number = _parse_number(answer)
        if answer_number is None:
            continue
        tolerance = 0.05 * max(abs(answer_number), 1.0)
        if math.isclose(prediction_number, answer_number, rel_tol=0.0, abs_tol=tolerance):
            return 1.0
    return 0.0


def score_prediction(metric_name: str, prediction: str, answers: list[str]) -> float:
    if metric_name in {"normalized_exact_match", "exact_match", "ocrbench_exact"}:
        return score_exact_match(prediction, answers)
    if metric_name == "anls":
        return score_anls(prediction, answers)
    if metric_name in {"numeric_relaxed_accuracy", "chart_numeric_relaxed"}:
        return score_numeric_relaxed_accuracy(prediction, answers)
    raise ValueError(f"Unsupported metric '{metric_name}'.")
