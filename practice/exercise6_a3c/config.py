from dataclasses import dataclass

from practice.exercise5_a2c.a2c_gae_exercise import A2CConfig


@dataclass(frozen=True, kw_only=True)
class A3CConfig(A2CConfig):
    """The configuration for the A3C algorithm."""

    num_workers: int
    """The number of workers."""
