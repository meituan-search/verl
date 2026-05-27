"""Constant reward function for GRPO benchmark runs.

Always returns 1.0 — rewards don't matter for a throughput test.
"""


def compute_score(solution_str: str, ground_truth: str, data_source: str = '', **kwargs) -> float:
    return 0.5
