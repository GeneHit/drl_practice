"""Schedule implementations for training parameters."""

import abc


class ScheduleBase(abc.ABC):
    """Base class for all schedules."""
    
    @abc.abstractmethod
    def __call__(self, t: int) -> float:
        """Get the value for the given time step."""
        pass


class ExponentialSchedule(ScheduleBase):
    """Exponential decay schedule."""
    
    def __init__(self, min_value: float, max_value: float, decay_rate: float) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.decay_rate = decay_rate
    
    def __call__(self, t: int) -> float:
        """Compute exponentially decayed value."""
        decayed = self.min_value + (self.max_value - self.min_value) * (self.decay_rate ** t)
        return max(decayed, self.min_value)


class LinearSchedule(ScheduleBase):
    """Linear decay schedule."""
    
    def __init__(self, min_value: float, max_value: float, duration: int) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.duration = duration
    
    def __call__(self, t: int) -> float:
        """Compute linearly decayed value."""
        if t >= self.duration:
            return self.min_value
        
        progress = t / self.duration
        return self.max_value - progress * (self.max_value - self.min_value)


class ConstantSchedule(ScheduleBase):
    """Constant value schedule."""
    
    def __init__(self, value: float) -> None:
        self.value = value
    
    def __call__(self, t: int) -> float:
        """Return constant value."""
        return self.value 