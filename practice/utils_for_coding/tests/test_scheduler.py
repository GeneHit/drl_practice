"""Tests for scheduler utilities."""

from typing import List, Tuple

import numpy as np
import pytest

from practice.utils_for_coding.scheduler_utils import (
    ExponentialSchedule,
    LinearSchedule,
)


class TestLinearSchedule:
    """Test cases for LinearSchedule."""

    @pytest.mark.parametrize(
        "start_e,end_e,duration,t,expected",
        [
            # Basic linear interpolation
            (1.0, 0.0, 100, 0, 1.0),  # Start value
            (1.0, 0.0, 100, 50, 0.5),  # Middle value
            (1.0, 0.0, 100, 100, 0.0),  # End value
            (
                1.0,
                0.0,
                100,
                150,
                0.0,
            ),  # Beyond duration (should clamp to end_e)
            # Increasing schedule
            (0.1, 0.9, 100, 0, 0.1),  # Start value
            (0.1, 0.9, 100, 25, 0.3),  # Quarter way
            (0.1, 0.9, 100, 100, 0.9),  # End value
            # Same start and end values
            (0.5, 0.5, 100, 50, 0.5),  # Constant schedule
            # Short duration
            (1.0, 0.0, 1, 0, 1.0),  # Start
            (1.0, 0.0, 1, 1, 0.0),  # End
        ],
    )
    def test_linear_schedule_values(
        self,
        start_e: float,
        end_e: float,
        duration: int,
        t: int,
        expected: float,
    ) -> None:
        """Test LinearSchedule returns correct values at different time steps."""
        schedule = LinearSchedule(v0=start_e, v1=end_e, t1=duration)
        result = schedule(t)
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    @pytest.mark.parametrize(
        "start_e,end_e,duration,time_steps,expected_values",
        [
            # Decreasing schedule over 10 steps
            (
                1.0,
                0.0,
                10,
                [0, 1, 2, 5, 10, 15],
                [1.0, 0.9, 0.8, 0.5, 0.0, 0.0],
            ),
            # Increasing schedule over 4 steps
            (0.2, 0.6, 4, [0, 1, 2, 4], [0.2, 0.3, 0.4, 0.6]),
        ],
    )
    def test_linear_schedule_sequence(
        self,
        start_e: float,
        end_e: float,
        duration: int,
        time_steps: List[int],
        expected_values: List[float],
    ) -> None:
        """Test LinearSchedule over a sequence of time steps."""
        schedule = LinearSchedule(v0=start_e, v1=end_e, t1=duration)

        for t, expected in zip(time_steps, expected_values):
            result = schedule(t)
            assert abs(result - expected) < 1e-6, f"At t={t}, expected {expected}, got {result}"

    @pytest.mark.parametrize(
        "v0,v1,duration,schedule_type",
        [
            (1.0, 0.1, 100, "decreasing"),
            (0.1, 1.0, 100, "increasing"),
            (0.5, 0.2, 50, "decreasing"),
            (0.2, 0.8, 50, "increasing"),
        ],
    )
    def test_linear_schedule_properties(
        self, v0: float, v1: float, duration: int, schedule_type: str
    ) -> None:
        """Test LinearSchedule properties and edge cases for both decreasing and increasing schedules."""
        schedule = LinearSchedule(v0=v0, v1=v1, t1=duration)

        # Test that the schedule is monotonic
        values = [schedule(t) for t in range(0, duration + 1, max(1, duration // 10))]

        if schedule_type == "decreasing":
            # For decreasing schedules, values should be non-increasing
            for i in range(1, len(values)):
                assert values[i] <= values[i - 1], (
                    f"Decreasing schedule should be non-increasing, but {values[i]} > {values[i - 1]} at step {i}"
                )

            # Test that values never go below end_e
            for t in range(0, duration * 2, max(1, duration // 20)):
                assert schedule(t) >= v1, f"Value at t={t} should not be below end_e={v1}"

        elif schedule_type == "increasing":
            # For increasing schedules, values should be non-decreasing
            for i in range(1, len(values)):
                assert values[i] >= values[i - 1], (
                    f"Increasing schedule should be non-decreasing, but {values[i]} < {values[i - 1]} at step {i}"
                )

            # Test that values never go above end_e (after duration)
            for t in range(duration, duration * 2, max(1, duration // 20)):
                assert schedule(t) <= v1 + 1e-10, f"Value at t={t} should not exceed end_e={v1}"

        # Test boundary values
        assert abs(schedule(0) - v0) < 1e-10, f"At t=0, should return start_e={v0}"
        assert abs(schedule(duration) - v1) < 1e-10, f"At t=duration, should return end_e={v1}"
        assert abs(schedule(duration + 50) - v1) < 1e-10, (
            f"Beyond duration, should return end_e={v1}"
        )


class TestExponentialSchedule:
    """Test cases for ExponentialSchedule."""

    @pytest.mark.parametrize(
        "start_e,end_e,decay_rate,t,tolerance",
        [
            # Basic exponential decay
            (1.0, 0.1, 0.01, 0, 1e-6),  # Start value should be start_e
            (1.0, 0.1, 0.01, 100, 1e-6),  # At t=100, should approach some value
            (1.0, 0.1, 0.01, 200, 1e-6),  # At t=200, should be closer to end_e
            # Different parameters
            (0.8, 0.2, 0.02, 0, 1e-6),  # Different start/end values
            (0.5, 0.05, 0.005, 0, 1e-6),  # Different decay rate
        ],
    )
    def test_exponential_schedule_start_value(
        self,
        start_e: float,
        end_e: float,
        decay_rate: float,
        t: int,
        tolerance: float,
    ) -> None:
        """Test ExponentialSchedule returns correct start value."""
        schedule = ExponentialSchedule(v0=start_e, v1=end_e, decay_rate=decay_rate)
        if t == 0:
            result = schedule(t)
            assert abs(result - start_e) < tolerance, f"At t=0, expected {start_e}, got {result}"

    @pytest.mark.parametrize(
        "start_e,end_e,decay_rate",
        [
            (1.0, 0.1, 0.01),
            (0.8, 0.2, 0.02),
            (0.5, 0.05, 0.005),
        ],
    )
    def test_exponential_schedule_monotonic(
        self, start_e: float, end_e: float, decay_rate: float
    ) -> None:
        """Test that ExponentialSchedule is monotonically decreasing when start_e > end_e."""
        schedule = ExponentialSchedule(v0=start_e, v1=end_e, decay_rate=decay_rate)

        # Sample points throughout the schedule
        time_points = list(range(0, 200, 10))
        values = [schedule(t) for t in time_points]

        if start_e > end_e:
            # Check monotonic decreasing property
            for i in range(1, len(values)):
                assert values[i] <= values[i - 1] + 1e-10, (
                    f"Schedule should be non-increasing, but value increased from {values[i - 1]} to {values[i]} at step {i}"
                )

    @pytest.mark.parametrize(
        "start_e,end_e,decay_rate,test_points",
        [
            (1.0, 0.1, 0.01, [0, 10, 25, 50, 100, 150, 200]),
            (0.8, 0.2, 0.02, [0, 5, 12, 25, 50, 75]),
        ],
    )
    def test_exponential_schedule_bounds(
        self,
        start_e: float,
        end_e: float,
        decay_rate: float,
        test_points: List[int],
    ) -> None:
        """Test that ExponentialSchedule respects bounds."""
        schedule = ExponentialSchedule(v0=start_e, v1=end_e, decay_rate=decay_rate)

        for t in test_points:
            result = schedule(t)
            if start_e > end_e:
                # For decreasing schedules
                assert result >= end_e - 1e-10, (
                    f"At t={t}, value {result} should not be below end_e={end_e}"
                )
                assert result <= start_e + 1e-10, (
                    f"At t={t}, value {result} should not exceed start_e={start_e}"
                )

    def test_exponential_schedule_implementation(self) -> None:
        """Test the mathematical correctness of ExponentialSchedule implementation."""
        start_e, end_e, decay_rate = 1.0, 0.1, 0.01
        schedule = ExponentialSchedule(v0=start_e, v1=end_e, decay_rate=decay_rate)

        # Test specific mathematical property - exponential decay
        t = 50
        result = schedule(t)

        # Calculate expected value using the correct exponential decay formula
        # epsilon = end_e + (start_e - end_e) * exp(-decay_rate * t)
        expected = end_e + (start_e - end_e) * np.exp(-decay_rate * t)

        assert abs(result - expected) < 1e-10, (
            f"Mathematical implementation incorrect: expected {expected}, got {result}"
        )

    def test_exponential_schedule_type_annotations(self) -> None:
        """Test that ExponentialSchedule properly handles type annotations."""
        schedule = ExponentialSchedule(v0=1.0, v1=0.1, decay_rate=0.01)
        result: float = schedule(50)
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "schedule_params,expected_type",
        [
            ((1.0, 0.1, 0.01), float),
            ((0.5, 0.05, 0.005), float),
        ],
    )
    def test_schedule_return_types(
        self, schedule_params: Tuple[float, float, float], expected_type: type
    ) -> None:
        """Test that schedules return the correct type."""
        start_e, end_e, decay_rate = schedule_params

        linear_schedule = LinearSchedule(v0=start_e, v1=end_e, t1=100)
        exp_schedule = ExponentialSchedule(v0=start_e, v1=end_e, decay_rate=decay_rate)

        linear_result = linear_schedule(25)
        exp_result = exp_schedule(25)

        assert isinstance(linear_result, expected_type)
        assert isinstance(exp_result, expected_type)


class TestScheduleComparison:
    """Test cases comparing different schedule types."""

    @pytest.mark.parametrize(
        "start_e,end_e,duration,decay_rate",
        [
            (1.0, 0.1, 100, 0.01),
            (0.8, 0.2, 50, 0.02),
        ],
    )
    def test_schedule_start_end_consistency(
        self, start_e: float, end_e: float, duration: int, decay_rate: float
    ) -> None:
        """Test that both schedules have consistent start and end behavior."""
        linear_schedule = LinearSchedule(v0=start_e, v1=end_e, t1=duration)
        exp_schedule = ExponentialSchedule(v0=start_e, v1=end_e, decay_rate=decay_rate)

        # Both should start at start_e
        assert abs(linear_schedule(0) - start_e) < 1e-10
        assert abs(exp_schedule(0) - start_e) < 1e-10

        # Linear schedule should reach end_e at duration and beyond
        assert abs(linear_schedule(duration) - end_e) < 1e-10
        assert abs(linear_schedule(duration + 50) - end_e) < 1e-10

        # Exponential schedule should approach end_e as t increases
        # At large t values, it should be close to end_e
        if start_e > end_e:
            large_t_value = exp_schedule(1000)  # Very large t
            assert large_t_value >= end_e - 1e-6, "At large t, exp schedule should approach end_e"
            assert large_t_value <= start_e, "Exp schedule should not exceed start_e"
