import numpy as np

from common.base import ScheduleBase


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

    def __init__(self, start_e: float, end_e: float, duration: int) -> None:
        self.start_e = start_e
        self.end_e = end_e
        self.duration = duration

    def __call__(self, t: int) -> float:
        assert t >= 0
        if t >= self.duration:
            return self.end_e

        t_e = self.start_e + (self.end_e - self.start_e) * (t / self.duration)
        return float(t_e)


class ExponentialSchedule(ScheduleBase):
    """Exponential schedule.

    t=0: epsilon should be start_e
    t -> inf: epsilon should be end_e
    """

    def __init__(self, start_e: float, end_e: float, decay_rate: float) -> None:
        self.start_e = start_e
        self.end_e = end_e
        self.decay_rate = decay_rate

    def __call__(self, t: int) -> float:
        assert t >= 0
        # when t=0, epsilon should be start_e
        # when t -> inf, epsilon should be end_e
        epsilon = self.end_e + (self.start_e - self.end_e) * float(
            np.exp(-self.decay_rate * t)
        )
        return epsilon
