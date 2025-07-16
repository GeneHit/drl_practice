import pytest
import torch

from practice.utils_for_coding.baseline_utils import (
    ConstantBaseline,
    OptimalConstantBaseline,
    TimeDependentBaseline,
)


def test_constant_baseline() -> None:
    baseline = ConstantBaseline(decay=0.0)
    # Empty returns
    assert baseline.update([]) == 0.0
    # Typical returns
    assert baseline.update([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert baseline.update([10.0, -10.0]) == 0.0


def test_optimal_constant_baseline() -> None:
    baseline = OptimalConstantBaseline()
    # Empty returns
    assert baseline.update([], []) == 0.0
    # No log_probs
    assert baseline.update([1.0, 2.0], None) == 0.0
    # Typical case
    returns = [1.0, 2.0, 3.0]
    log_probs = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    gradsq = [1.0, 4.0, 9.0]
    expected = sum(g * r for g, r in zip(gradsq, returns)) / sum(gradsq)
    assert baseline.update(returns, log_probs) == pytest.approx(expected)


def test_time_dependent_baseline() -> None:
    baseline = TimeDependentBaseline()
    # Empty returns
    assert baseline.update([]) == []
    # Typical returns
    returns = [1.0, 2.0, 3.0]
    expected = [sum(returns[t:]) for t in range(len(returns))]
    assert baseline.update(returns) == expected
    returns = [10.0, -10.0, 5.0]
    expected = [5.0, -5.0, 5.0]
    assert baseline.update(returns) == expected
