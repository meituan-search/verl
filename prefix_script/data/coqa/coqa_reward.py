"""Reward function for CoQA (Conversational Question Answering).

Score = F1 overlap between normalized model answer and ground truth.
Returns 1.0 for exact match, partial credit for partial overlap, 0.0 for no overlap.

Usage in verl config:
    reward.custom_reward_function.path: ~/dataset/reward/coqa_reward.py
    reward.custom_reward_function.name: compute_score
"""

import re
import string


def _normalize(text: str) -> str:
    """Lowercase, remove punctuation/articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _extract_answer(response: str) -> str:
    """Extract the answer from model response.

    Tries (in order):
    1. Text after 'Answer:' or 'answer:'
    2. Last non-empty line
    3. Full response (trimmed)
    """
    for prefix in ("Answer:", "answer:", "ANSWER:"):
        if prefix in response:
            return response.split(prefix, 1)[1].strip().split("\n")[0].strip()
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    return lines[-1] if lines else response.strip()


def compute_score(
    solution_str: str,
    ground_truth: str,
    data_source: str = "coqa",
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Compute F1-based reward for CoQA.

    Args:
        solution_str: model's raw response text.
        ground_truth: correct answer string (from reward_model.ground_truth).
        data_source: ignored (always "coqa").
        extra_info: optional dict; if ground_truth is empty, falls back to
                    extra_info["answer"].

    Returns:
        float in [0, 1] — token-level F1 between extracted answer and gold.
    """
    gold = ground_truth
    if not gold and extra_info:
        gold = extra_info.get("answer", "")
    if not gold:
        return 0.0

    pred = _extract_answer(solution_str)
    return _f1(pred, gold)
