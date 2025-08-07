import abc
import math

import numpy as np


class ScheduleBase(abc.ABC):
    """Base class for all schedules."""

    @abc.abstractmethod
    def __call__(self, t: int) -> float:
        """Get the value for the given time.

        Usage:
        ```python
        schedule = LinearSchedule(min_e=0.01, max_e=1.0, duration=1000)
        epsilon = schedule(t=100)
        ```
        """


class ConstantSchedule(ScheduleBase):
    """Constant schedule."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, t: int) -> float:
        return self.value


class LinearSchedule(ScheduleBase):
    """Linear schedule."""

    def __init__(self, v0: float, v1: float, t1: int, t0: int = 0) -> None:
        """Linear schedule.

        output:
        1. if t <= t0: v = v0
        2. if t >= t1: v = v1
        3. if t0 < t < t1: v = v0 + (v1 - v0) * (t - t0) / (t1 - t0)

        Parameters
        ----------
            v0: The start value.
            v1: The end value.
            t0: The start time.
            t1: The end time.
        """
        self._v0 = v0
        self._v1 = v1
        self._t0 = t0
        self._t1 = t1
        self._duration = t1 - t0
        assert self._t0 <= self._t1
        self._v_diff = v1 - v0

    def __call__(self, t: int) -> float:
        assert t >= 0
        if t <= self._t0:
            return self._v0
        if t >= self._t1:
            return self._v1
        return self._v0 + self._v_diff * (t - self._t0) / self._duration


class ExponentialSchedule(ScheduleBase):
    """Exponential schedule.

    t=0: epsilon should be start_e
    t -> inf: epsilon should be end_e
    """

    def __init__(self, v0: float, v1: float, decay_rate: float) -> None:
        self._v0 = v0
        self._v1 = v1
        self._decay_rate = decay_rate
        self._v_diff = v0 - v1

    def __call__(self, t: int) -> float:
        assert t >= 0
        # when t=0, output v0
        # when t -> inf, output v1
        return self._v1 + self._v_diff * float(np.exp(-self._decay_rate * t))


class CosineSchedule(ScheduleBase):
    def __init__(
        self, v0: float, v1: float, t1: int, t0: int = 0, decay_factor: float = 1.0
    ) -> None:
        """Cosine decay from v0 to v1 with t0 -> t1.

        Parameters
        ----------
            v0: The start value.
            v1: The end value.
            t0: The start step.
            t1: The end step.
            decay_factor: The decay factor, controls the speed of decay (default 1.0 for standard cosine)
        """
        self._v0 = v0
        self._v1 = v1
        self._t0 = t0
        self._t1 = t1
        self._decay_factor = decay_factor
        self._duration = t1 - t0
        assert self._duration >= 0
        self._decay_factor = decay_factor
        self._v_diff = v0 - v1

    def __call__(self, t: int) -> float:
        if t <= self._t0:
            return self._v0
        elif t >= self._t1:
            return self._v1
        else:
            progress = (t - self._t0) / self._duration
            cosine_decay = 0.5 * (1 + math.cos(math.pi * self._decay_factor * progress))
            return self._v1 + self._v_diff * cosine_decay
