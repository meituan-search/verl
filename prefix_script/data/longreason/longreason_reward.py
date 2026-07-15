# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reward function for LongReason (long-context multiple-choice QA).

Extracts the letter answer from "The answer is X" in the model response
and compares it to the ground truth letter (A/B/C/D/E).

Returns 1.0 for correct, 0.0 for incorrect or unparseable.
"""

import re


def _extract_letter(response: str) -> str:
    """Extract the answer letter from model response.

    Looks for 'The answer is X' pattern (case-insensitive).
    Falls back to the last standalone A/B/C/D/E on a line.
    """
    # Primary: "The answer is X" pattern
    m = re.search(r"[Tt]he answer is\s*([A-Ea-e])", response)
    if m:
        return m.group(1).upper()
    # Fallback: last line containing only a letter option
    for line in reversed(response.strip().splitlines()):
        line = line.strip().rstrip(".")
        if line.upper() in ("A", "B", "C", "D", "E"):
            return line.upper()
    return ""


def compute_score(
    solution_str: str,
    ground_truth: str,
    data_source: str = "longreason",
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Exact-match reward for multiple-choice LongReason.

    Args:
        solution_str: model's raw response text.
        ground_truth: correct letter (A/B/C/D/E).
        data_source: ignored.
        extra_info: optional dict; falls back to extra_info["answer"] if ground_truth empty.

    Returns:
        1.0 if correct letter extracted, 0.0 otherwise.
    """
    gold = (ground_truth or "").strip().upper()
    if not gold and extra_info:
        gold = str(extra_info.get("answer", "")).strip().upper()
    if not gold:
        return 0.0

    pred = _extract_letter(solution_str)
    return 1.0 if pred == gold else 0.0
