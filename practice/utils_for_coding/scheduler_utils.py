import abc

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
    """Linear schedule.

    t = 0: epsilon = start_e
    t >= duration: epsilon = end_e
    """

    def __init__(
        self,
        start_e: float,
        end_e: float,
        duration: int,
        start_t: int = 0,
    ) -> None:
        self._start_e = start_e
        self._end_e = end_e
        self._duration = duration
        self._start_t = start_t
        assert self._start_t <= self._duration

    def __call__(self, t: int) -> float:
        assert t >= 0
        if t <= self._start_t:
            return self._start_e

        t = t - self._start_t
        if t >= self._duration:
            return self._end_e

        t_e = self._start_e + (self._end_e - self._start_e) * (t / self._duration)
        return t_e


class ExponentialSchedule(ScheduleBase):
    """Exponential schedule.

    t=0: epsilon should be start_e
    t -> inf: epsilon should be end_e
    """

    def __init__(self, start_e: float, end_e: float, decay_rate: float) -> None:
        self._start_e = start_e
        self._end_e = end_e
        self._decay_rate = decay_rate

    def __call__(self, t: int) -> float:
        assert t >= 0
        # when t=0, epsilon should be start_e
        # when t -> inf, epsilon should be end_e
        epsilon = self._end_e + (self._start_e - self._end_e) * float(np.exp(-self._decay_rate * t))
        return epsilon
