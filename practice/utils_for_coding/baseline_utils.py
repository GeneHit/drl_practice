import abc
from typing import List, Optional, Union

from torch import Tensor


class BaselineBase(abc.ABC):
    """Base class for all baseline strategies for variance reduction in policy gradient methods."""

    @abc.abstractmethod
    def update(
        self, returns: List[float], log_probs: Optional[List[Tensor]] = None
    ) -> Union[float, List[float]]:
        """Update the baseline with new episode returns and (optionally) log_probs, and return the baseline value(s)."""


class ConstantBaseline(BaselineBase):
    """Constant baseline: b = mean of returns."""

    def __init__(self, decay: float = 0.99) -> None:
        self._decay = decay
        self._baseline_value = 0.0
        self._baseline_initialized = False

    def update(self, returns: List[float], log_probs: Optional[List[Tensor]] = None) -> float:
        mean_return = sum(returns) / len(returns)
        if not self._baseline_initialized:
            self._baseline_value = mean_return
            self._baseline_initialized = True
        else:
            self._baseline_value = (
                self._decay * self._baseline_value + (1 - self._decay) * mean_return
            )
        return self._baseline_value


class OptimalConstantBaseline(BaselineBase):
    """Optimal constant baseline: weighted by grad log prob squared."""

    def update(self, returns: List[float], log_probs: Optional[List[Tensor]] = None) -> float:
        if log_probs is None or not returns:
            return 0.0
        gradsq = [float(lp.detach().cpu().item()) ** 2 for lp in log_probs]
        numerator = sum(g * r for g, r in zip(gradsq, returns))
        denominator = sum(gradsq) if sum(gradsq) != 0 else 1.0
        return numerator / denominator


class TimeDependentBaseline(BaselineBase):
    """Time-dependent baseline: b_t = mean of returns from t to end."""

    def update(self, returns: List[float], log_probs: Optional[List[Tensor]] = None) -> List[float]:
        H = len(returns)
        return [sum(returns[t:]) for t in range(H)]
