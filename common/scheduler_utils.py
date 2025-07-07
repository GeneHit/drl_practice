import numpy as np

from common.base import ScheduleBase


class ConstantSchedule(ScheduleBase):
    """Constant schedule."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, t: int) -> float:
        return self.value


class LinearSchedule(ScheduleBase):
    """Linear schedule."""

    def __init__(self, start_e: float, end_e: float, duration: int) -> None:
        self.start_e = start_e
        self.end_e = end_e
        self.duration = duration

    def __call__(self, t: int) -> float:
        t_e = self.start_e + (self.end_e - self.start_e) * (t / self.duration)
        return float(max(t_e, self.end_e))


class ExponentialSchedule(ScheduleBase):
    """Exponential schedule."""

    def __init__(self, start_e: float, end_e: float, duration: int) -> None:
        self.start_e = start_e
        self.end_e = end_e
        self.duration = duration

    def __call__(self, t: int) -> float:
        t_e = (
            self.start_e
            * (self.end_e - self.start_e)
            * np.exp(t / self.duration)
        )
        return float(max(t_e, self.end_e))
