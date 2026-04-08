from __future__ import annotations

import ast
import math
import re


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s.%/-]+", re.UNICODE)
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
_OPTION_RE = re.compile(r"\b([A-G])\b", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|answer)\s*[:：]?\s*[\(\[]?\s*([A-G])\s*[\)\]]?",
    re.IGNORECASE,
)


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
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    token = matches[-1].replace(",", "").replace("$", "").strip()
    if token.endswith("%"):
        return float(token[:-1])
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


def _coerce_options(metadata: dict | None) -> list[str]:
    metadata = metadata or {}
    raw = metadata.get("options")
    if raw is None:
        raw = metadata.get("choices")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            pass
    return []


def _extract_option_letter(prediction: str) -> str | None:
    final_matches = _FINAL_ANSWER_RE.findall(prediction)
    if final_matches:
        return final_matches[-1].upper()

    stripped = prediction.strip()
    exact_match = re.fullmatch(r"[\(\[]?\s*([A-G])\s*[\)\].:]?\s*$", stripped, flags=re.IGNORECASE)
    if exact_match:
        return exact_match.group(1).upper()

    line_matches = re.findall(r"^[\-\*\s]*([A-G])[\.\)]\s", prediction, flags=re.IGNORECASE | re.MULTILINE)
    if line_matches:
        return line_matches[-1].upper()
    return None


def _match_option_text(prediction: str, options: list[str]) -> str | None:
    normalized_prediction = normalize_text(prediction)
    for index, option in enumerate(options):
        normalized_option = normalize_text(option)
        if normalized_option and normalized_prediction == normalized_option:
            return chr(65 + index)
    return None


def score_ai2d_option_match(prediction: str, answers: list[str], metadata: dict | None) -> float:
    if not answers:
        return 0.0
    options = _coerce_options(metadata)
    try:
        answer_index = int(str(answers[0]))
    except ValueError:
        return score_exact_match(prediction, answers)

    correct_letter = chr(65 + answer_index)
    correct_option = options[answer_index] if 0 <= answer_index < len(options) else None
    predicted_letter = _extract_option_letter(prediction) or _match_option_text(prediction, options)
    if predicted_letter == correct_letter:
        return 1.0
    if correct_option and normalize_text(correct_option) in normalize_text(prediction):
        return 1.0
    if normalize_text(prediction) == normalize_text(str(answer_index)):
        return 1.0
    return 0.0


def score_mmmu_option_match(prediction: str, answers: list[str], metadata: dict | None) -> float:
    if not answers:
        return 0.0
    correct_letter = normalize_text(answers[0]).upper()
    options = _coerce_options(metadata)
    predicted_letter = _extract_option_letter(prediction) or _match_option_text(prediction, options)
    if predicted_letter and predicted_letter.upper() == correct_letter:
        return 1.0

    if options:
        answer_index = ord(correct_letter) - 65
        if 0 <= answer_index < len(options):
            if normalize_text(options[answer_index]) in normalize_text(prediction):
                return 1.0
    return 0.0


def score_mathvista_match(prediction: str, answers: list[str], metadata: dict | None) -> float:
    metadata = metadata or {}
    answer_type = str(metadata.get("answer_type") or "").lower()
    question_type = str(metadata.get("question_type") or "").lower()
    options = _coerce_options(metadata)

    if options or "multi" in question_type:
        correct_letters: set[str] = set()
        normalized_answers = {normalize_text(answer) for answer in answers}
        for answer in answers:
            normalized = normalize_text(answer).upper()
            if re.fullmatch(r"[A-G]", normalized):
                correct_letters.add(normalized)
        for index, option in enumerate(options):
            if normalize_text(option) in normalized_answers:
                correct_letters.add(chr(65 + index))

        predicted_letter = _extract_option_letter(prediction)
        if predicted_letter is None:
            predicted_letter = _match_option_text(prediction, options)
        if predicted_letter and predicted_letter in correct_letters:
            return 1.0

        normalized_prediction = normalize_text(prediction)
        for index, option in enumerate(options):
            if normalized_prediction == normalize_text(option) and chr(65 + index) in correct_letters:
                return 1.0
        return 0.0

    if answer_type in {"float", "integer"}:
        prediction_number = _parse_number(prediction)
        if prediction_number is None:
            return 0.0
        for answer in answers:
            answer_number = _parse_number(answer)
            if answer_number is None:
                continue
            if answer_type == "integer":
                return 1.0 if int(round(prediction_number)) == int(round(answer_number)) else 0.0
            precision = metadata.get("precision")
            tolerance = 10 ** (-int(precision)) if precision not in (None, "", "None") else 1e-3
            if math.isclose(prediction_number, answer_number, rel_tol=0.0, abs_tol=tolerance):
                return 1.0
        return 0.0

    return score_exact_match(prediction, answers)


def score_prediction(
    metric_name: str,
    prediction: str,
    answers: list[str],
    metadata: dict | None = None,
) -> float:
    if metric_name in {"normalized_exact_match", "exact_match", "ocrbench_exact"}:
        return score_exact_match(prediction, answers)
    if metric_name == "anls":
        return score_anls(prediction, answers)
    if metric_name in {"numeric_relaxed_accuracy", "chart_numeric_relaxed"}:
        return score_numeric_relaxed_accuracy(prediction, answers)
    if metric_name == "ai2d_option_match":
        return score_ai2d_option_match(prediction, answers, metadata)
    if metric_name == "mmmu_option_match":
        return score_mmmu_option_match(prediction, answers, metadata)
    if metric_name == "mathvista_match":
        return score_mathvista_match(prediction, answers, metadata)
    raise ValueError(f"Unsupported metric '{metric_name}'.")
