"""Statistical tests for comparing RL agents vs. baselines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class TestResult:
    """Result of a statistical comparison."""

    statistic: float
    p_value: float
    effect_size: float
    significant: bool
    test_name: str


def mann_whitney_u(
    group_a: NDArray[np.float64],
    group_b: NDArray[np.float64],
    alpha: float = 0.05,
) -> TestResult:
    """Perform Mann-Whitney U test (non-parametric).

    Args:
        group_a: KPI values for group A (e.g., baseline).
        group_b: KPI values for group B (e.g., RL agent).
        alpha: Significance level.

    Returns:
        TestResult with U statistic, p-value, and effect size.
    """
    stat, p = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    r = compute_effect_size(group_a, group_b)
    return TestResult(
        statistic=stat,
        p_value=p,
        effect_size=r,
        significant=p < alpha,
        test_name="Mann-Whitney U",
    )


def compute_effect_size(
    group_a: NDArray[np.float64],
    group_b: NDArray[np.float64],
) -> float:
    """Compute rank-biserial correlation as effect size for Mann-Whitney U.

    Args:
        group_a: First group values.
        group_b: Second group values.

    Returns:
        Effect size r in [-1, 1].
    """
    n1, n2 = len(group_a), len(group_b)
    stat, _ = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    # Rank-biserial correlation: r = 1 - (2U)/(n1*n2)
    return 1.0 - (2.0 * stat) / (n1 * n2)
