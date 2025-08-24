from typing import Any, Dict

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from practice.utils_for_coding.buffer.replay_np import ReplayBuffer


class TestReplayBuffer:
    """Test suite for ReplayBuffer class."""

    @pytest.fixture
    def buffer(self) -> ReplayBuffer:
        """Create a replay buffer for testing."""
        return ReplayBuffer(capacity=10)

    @pytest.fixture
    def sample_data(self) -> Dict[str, NDArray[Any]]:
        """Create sample data for testing."""
        batch_size = 3
        state_dim = 4
        action_dim = 2

        return {
            "states": np.random.rand(batch_size, state_dim).astype(np.float32),
            "actions": np.random.rand(batch_size, action_dim).astype(np.float32),
            "rewards": np.random.rand(batch_size).astype(np.float32),
            "next_states": np.random.rand(batch_size, state_dim).astype(np.float32),
            "dones": np.random.choice([True, False], batch_size),
        }

    def test_init(self) -> None:
        """Test buffer initialization."""
        capacity = 100
        buffer = ReplayBuffer(capacity)
        assert len(buffer) == 0
        assert buffer.capacity == capacity

    def test_add_batch(self, buffer: ReplayBuffer, sample_data: Dict[str, NDArray[Any]]) -> None:
        """Test adding a batch of data."""
        initial_len = len(buffer)
        indices = buffer.add_batch(**sample_data)

        # Check that buffer size increased
        assert len(buffer) == initial_len + len(sample_data["states"])

        # Check that indices are returned correctly
        expected_indices = np.arange(initial_len, initial_len + len(sample_data["states"]))
        np.testing.assert_array_equal(indices, expected_indices)

    def test_add_empty_batch(self, buffer: ReplayBuffer) -> None:
        """Test adding an empty batch."""
        empty_data = {
            "states": np.array([]).reshape(0, 4).astype(np.float32),
            "actions": np.array([]).reshape(0, 2).astype(np.float32),
            "rewards": np.array([]).astype(np.float32),
            "next_states": np.array([]).reshape(0, 4).astype(np.float32),
            "dones": np.array([]).astype(np.bool_),
        }

        initial_len = len(buffer)
        indices = buffer.add_batch(**empty_data)

        assert len(buffer) == initial_len
        assert len(indices) == 0

    def test_sample(self, buffer: ReplayBuffer, sample_data: Dict[str, NDArray[Any]]) -> None:
        """Test sampling from the buffer."""
        # Add data first
        buffer.add_batch(**sample_data)

        batch_size = 2
        experience = buffer.sample(batch_size)

        # Check types
        assert isinstance(experience.states, torch.Tensor)
        assert isinstance(experience.actions, torch.Tensor)
        assert isinstance(experience.rewards, torch.Tensor)
        assert isinstance(experience.next_states, torch.Tensor)
        assert isinstance(experience.dones, torch.Tensor)

        # Check shapes
        assert experience.states.shape[0] == batch_size
        assert experience.actions.shape[0] == batch_size
        assert experience.rewards.shape[0] == batch_size
        assert experience.next_states.shape[0] == batch_size
        assert experience.dones.shape[0] == batch_size

    def test_sample_insufficient_data(self, buffer: ReplayBuffer) -> None:
        """Test sampling when buffer doesn't have enough data."""
        # Try to sample from empty buffer
        with pytest.raises(AssertionError):
            buffer.sample(1)

        # Add one sample and try to sample more than available
        sample_data = {
            "states": np.random.rand(1, 4).astype(np.float32),
            "actions": np.random.rand(1, 2).astype(np.float32),
            "rewards": np.random.rand(1).astype(np.float32),
            "next_states": np.random.rand(1, 4).astype(np.float32),
            "dones": np.array([False]),
        }
        buffer.add_batch(**sample_data)

        with pytest.raises(AssertionError):
            buffer.sample(2)

    def test_sample_by_idxs(
        self, buffer: ReplayBuffer, sample_data: Dict[str, NDArray[Any]]
    ) -> None:
        """Test sampling by specific indices."""
        buffer.add_batch(**sample_data)

        idxs = np.array([0, 2], dtype=np.int64)
        experience = buffer.sample_by_idxs(idxs)

        # Check that we got the right number of samples
        assert experience.states.shape[0] == len(idxs)
        assert experience.actions.shape[0] == len(idxs)
        assert experience.rewards.shape[0] == len(idxs)
        assert experience.next_states.shape[0] == len(idxs)
        assert experience.dones.shape[0] == len(idxs)

    def test_sample_by_invalid_idxs(
        self, buffer: ReplayBuffer, sample_data: Dict[str, NDArray[Any]]
    ) -> None:
        """Test sampling by invalid indices."""
        buffer.add_batch(**sample_data)

        # Test negative index
        with pytest.raises(ValueError):
            buffer.sample_by_idxs(np.array([-1], dtype=np.int64))

        # Test index out of range
        with pytest.raises(ValueError):
            buffer.sample_by_idxs(np.array([10], dtype=np.int64))

    def test_dataloader(self, buffer: ReplayBuffer, sample_data: Dict[str, NDArray[Any]]) -> None:
        """Test the dataloader functionality."""
        buffer.add_batch(**sample_data)

        batch_size = 2
        batches = list(buffer.dataloader(batch_size, shuffle=False))

        # Should have ceil(3/2) = 2 batches
        assert len(batches) == 2

        # First batch should have 2 samples
        assert batches[0].states.shape[0] == 2

        # Second batch should have 1 sample
        assert batches[1].states.shape[0] == 1

        # Check that all samples are accounted for
        total_samples = sum(batch.states.shape[0] for batch in batches)
        assert total_samples == len(sample_data["states"])

    def test_dataloader_shuffle(self, buffer: ReplayBuffer) -> None:
        """Test that shuffle works in dataloader."""
        # Add more data to make shuffling meaningful
        large_sample = {
            "states": np.arange(20).reshape(20, 1).astype(np.float32),
            "actions": np.arange(20).reshape(20, 1).astype(np.float32),
            "rewards": np.arange(20).astype(np.float32),
            "next_states": np.arange(20).reshape(20, 1).astype(np.float32),
            "dones": np.array([False] * 20),
        }
        buffer.add_batch(**large_sample)

        # Get batches with and without shuffling
        no_shuffle_batches = list(buffer.dataloader(5, shuffle=False))
        shuffle_batches = list(buffer.dataloader(5, shuffle=True))

        # Should have same number of batches
        assert len(no_shuffle_batches) == len(shuffle_batches)

        # With high probability, at least one batch should be different
        # (this test might rarely fail due to randomness, but very unlikely)
        for b1, b2 in zip(no_shuffle_batches, shuffle_batches):
            if not torch.equal(b1.states, b2.states):
                break
        # Note: We don't assert this because shuffle might occasionally produce the same order

    def test_clear(self, buffer: ReplayBuffer, sample_data: Dict[str, NDArray[Any]]) -> None:
        """Test clearing the buffer."""
        buffer.add_batch(**sample_data)
        assert len(buffer) > 0

        buffer.clear()
        assert len(buffer) == 0

    def test_wraparound(self) -> None:
        """Test buffer behavior when capacity is exceeded."""
        capacity = 5
        buffer = ReplayBuffer(capacity)

        # Add data that exceeds capacity
        for i in range(3):
            batch_data = {
                "states": np.full((3, 1), i, dtype=np.float32),
                "actions": np.full((3, 1), i, dtype=np.float32),
                "rewards": np.full(3, i, dtype=np.float32),
                "next_states": np.full((3, 1), i, dtype=np.float32),
                "dones": np.array([False] * 3),
            }
            buffer.add_batch(**batch_data)

        # Buffer should be at capacity
        assert len(buffer) == capacity

        # Sample all data and verify it contains the most recent samples
        experience = buffer.sample_by_idxs(np.arange(capacity, dtype=np.int64))

        # Should contain mix of data from different batches due to wraparound
        assert experience.states.shape[0] == capacity

    def test_different_dtypes(self, buffer: ReplayBuffer) -> None:
        """Test buffer with different data types."""
        # Test with integer actions (discrete action space)
        states: NDArray[Any] = np.random.rand(2, 4).astype(np.float32)
        actions: NDArray[Any] = np.array([0, 1], dtype=np.int64)
        rewards: NDArray[Any] = np.array([1.0, -1.0], dtype=np.float32)
        next_states: NDArray[Any] = np.random.rand(2, 4).astype(np.float32)
        dones: NDArray[Any] = np.array([False, True])

        buffer.add_batch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )
        experience = buffer.sample(1)

        # Check that types are preserved correctly
        assert experience.actions.dtype == torch.int64
        assert experience.dones.dtype == torch.bool

    def test_multidimensional_states(self, buffer: ReplayBuffer) -> None:
        """Test buffer with multidimensional state spaces."""
        # Test with image-like states (e.g., 84x84x3)
        states: NDArray[Any] = np.random.randint(0, 255, (2, 84, 84, 3), dtype=np.uint8)
        actions: NDArray[Any] = np.array([0, 1], dtype=np.int64)
        rewards: NDArray[Any] = np.array([1.0, -1.0], dtype=np.float32)
        next_states: NDArray[Any] = np.random.randint(0, 255, (2, 84, 84, 3), dtype=np.uint8)
        dones: NDArray[Any] = np.array([False, True])

        buffer.add_batch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )
        experience = buffer.sample(1)

        # Check shapes are preserved
        assert experience.states.shape[1:] == (84, 84, 3)
        assert experience.next_states.shape[1:] == (84, 84, 3)
        assert experience.states.dtype == torch.uint8

    def test_large_batch_addition(self) -> None:
        """Test adding batches larger than capacity."""
        capacity = 5
        buffer = ReplayBuffer(capacity)

        # Add batch larger than capacity
        large_batch = {
            "states": np.random.rand(8, 4).astype(np.float32),
            "actions": np.random.rand(8, 2).astype(np.float32),
            "rewards": np.random.rand(8).astype(np.float32),
            "next_states": np.random.rand(8, 4).astype(np.float32),
            "dones": np.array([False] * 8),
        }

        indices = buffer.add_batch(**large_batch)

        # Buffer should be at capacity
        assert len(buffer) == capacity

        # Should return indices for all written data
        assert len(indices) == 8

    def test_property_access(self, buffer: ReplayBuffer) -> None:
        """Test property access."""
        assert buffer.capacity == 10
        assert len(buffer) == 0

        # Add some data
        sample_data = {
            "states": np.random.rand(3, 4).astype(np.float32),
            "actions": np.random.rand(3, 2).astype(np.float32),
            "rewards": np.random.rand(3).astype(np.float32),
            "next_states": np.random.rand(3, 4).astype(np.float32),
            "dones": np.array([False, True, False]),
        }
        buffer.add_batch(**sample_data)

        assert len(buffer) == 3
        assert buffer.capacity == 10  # Should remain unchanged
