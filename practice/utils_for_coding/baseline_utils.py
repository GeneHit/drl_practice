import abc
from collections.abc import Sequence

from torch import Tensor


class BaselineBase(abc.ABC):
    """Base class for all baseline strategies for variance reduction in policy gradient methods."""

    @abc.abstractmethod
    def update(
        self, returns: Sequence[float], log_probs: Sequence[Tensor] | None = None
    ) -> list[float]:
        """Compute the baseline value(s).

        Update the baseline with new episode returns and (optionally) log_probs, and
        return the baseline value(s).

        Args:
            returns: The returns of the episode.
            log_probs: The log probabilities of the actions.

        Returns:
            list[float]: The baseline value(s), length is the same as the number of returns.
        """


class ConstantBaseline(BaselineBase):
    """Constant baseline: b = mean of returns.

    Same with the MovingAverageBaseline.
    """

    def __init__(self, decay: float = 0.99) -> None:
        self._decay = decay
        self._baseline_value = 0.0
        self._baseline_initialized = False

    def update(
        self, returns: Sequence[float], log_probs: Sequence[Tensor] | None = None
    ) -> list[float]:
        if not returns:
            return []
        mean_return = sum(returns) / len(returns)
        if not self._baseline_initialized:
            self._baseline_value = mean_return
            self._baseline_initialized = True
        else:
            self._baseline_value = (
                self._decay * self._baseline_value + (1 - self._decay) * mean_return
            )
        return [self._baseline_value] * len(returns)


class OptimalConstantBaseline(BaselineBase):
    """Optimal constant baseline: weighted by grad log prob squared."""

    def update(
        self, returns: Sequence[float], log_probs: Sequence[Tensor] | None = None
    ) -> list[float]:
        if not returns:
            return []
        if log_probs is None:
            return [0.0] * len(returns)
        gradsq = [float(lp.detach().cpu().item()) ** 2 for lp in log_probs]
        numerator = sum(g * r for g, r in zip(gradsq, returns))
        denominator = sum(gradsq) if sum(gradsq) != 0 else 1.0
        return [numerator / denominator] * len(returns)


class TimeDependentBaseline(BaselineBase):
    """Time-dependent baseline: b_t = mean of returns from t to end."""

    def update(
        self, returns: Sequence[float], log_probs: Sequence[Tensor] | None = None
    ) -> list[float]:
        H = len(returns)
        return [sum(returns[t:]) for t in range(H)]
