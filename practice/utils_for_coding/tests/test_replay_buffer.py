import numpy as np
import pytest
import torch

from practice.utils_for_coding.replay_buffer_utils import Buffer, Experience, ReplayBuffer


class TestBuffer:
    """Test cases for the Buffer class."""

    def test_init(self) -> None:
        """Test Buffer initialization."""
        capacity = 100
        buffer = Buffer(capacity)

        assert buffer._capacity == capacity
        assert buffer._ptr == 0
        assert buffer._size == 0
        assert buffer._data == {}
        assert not buffer._initialized

    def test_add_batch_empty(self) -> None:
        """Test adding empty batch."""
        buffer = Buffer(10)
        result = buffer.add_batch(states=np.array([]))

        assert len(result) == 0
        assert buffer._size == 0
        assert not buffer._initialized

    def test_add_batch_first_call(self) -> None:
        """Test first call to add_batch initializes memory."""
        buffer = Buffer(10)
        states = np.random.randn(3, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (3,)).astype(np.int64)
        rewards = np.random.randn(3).astype(np.float32)

        result = buffer.add_batch(states=states, actions=actions, rewards=rewards)

        assert buffer._initialized
        assert len(result) == 3
        assert buffer._size == 3
        assert buffer._ptr == 3
        assert "states" in buffer._data
        assert "actions" in buffer._data
        assert "rewards" in buffer._data

    def test_add_batch_no_wrap_around(self) -> None:
        """Test adding batch without wrap-around."""
        buffer = Buffer(10)
        states = np.random.randn(3, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (3,)).astype(np.int64)

        result = buffer.add_batch(states=states, actions=actions)

        assert len(result) == 3
        assert buffer._size == 3
        assert buffer._ptr == 3
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))

    def test_add_batch_with_wrap_around(self) -> None:
        """Test adding batch with wrap-around."""
        buffer = Buffer(5)
        states = np.random.randn(3, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (3,)).astype(np.int64)

        # Fill buffer to capacity
        buffer.add_batch(states=states, actions=actions)
        buffer.add_batch(states=states, actions=actions)

        # Add one more batch to trigger wrap-around
        result = buffer.add_batch(states=states, actions=actions)

        assert len(result) == 3
        assert buffer._size == 5  # Should be at capacity
        assert buffer._ptr == 4  # Should wrap around to 4 (3+3+3 = 9, 9 % 5 = 4)

    def test_sample(self) -> None:
        """Test sampling from buffer."""
        buffer = Buffer(10)
        states = np.random.randn(5, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (5,)).astype(np.int64)

        buffer.add_batch(states=states, actions=actions)

        # Test sampling
        sample = buffer.sample(3)

        assert isinstance(sample, dict)
        assert "states" in sample
        assert "actions" in sample
        assert sample["states"].shape == (3, 4)
        assert sample["actions"].shape == (3,)
        assert isinstance(sample["states"], torch.Tensor)
        assert isinstance(sample["actions"], torch.Tensor)

    def test_sample_by_idxs(self) -> None:
        """Test sampling by specific indices."""
        buffer = Buffer(10)
        states = np.random.randn(5, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (5,)).astype(np.int64)

        buffer.add_batch(states=states, actions=actions)

        # Test sampling by specific indices
        indices = np.array([0, 2, 4])
        sample = buffer.sample_by_idxs(indices)

        assert isinstance(sample, dict)
        assert "states" in sample
        assert "actions" in sample
        assert sample["states"].shape == (3, 4)
        assert sample["actions"].shape == (3,)

    def test_sample_uninitialized_error(self) -> None:
        """Test error when sampling from uninitialized buffer."""
        buffer = Buffer(10)

        with pytest.raises(ValueError, match="high <= 0"):
            buffer.sample(3)

    def test_sample_by_idxs_uninitialized_error(self) -> None:
        """Test error when sampling by indices from uninitialized buffer."""
        buffer = Buffer(10)

        with pytest.raises(AssertionError, match="Memory not initialized"):
            buffer.sample_by_idxs(np.array([0, 1]))

    def test_sample_size_error(self) -> None:
        """Test error when sampling more than buffer size."""
        buffer = Buffer(10)
        states = np.random.randn(3, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (3,)).astype(np.int64)

        buffer.add_batch(states=states, actions=actions)

        with pytest.raises(AssertionError):
            buffer.sample(5)

    def test_sample_by_idxs_size_error(self) -> None:
        """Test error when sampling by indices with invalid size."""
        buffer = Buffer(10)
        states = np.random.randn(3, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (3,)).astype(np.int64)

        buffer.add_batch(states=states, actions=actions)

        with pytest.raises(AssertionError):
            buffer.sample_by_idxs(np.array([0, 1, 2, 3, 4]))

    def test_len(self) -> None:
        """Test buffer length."""
        buffer = Buffer(10)
        assert len(buffer) == 0

        states = np.random.randn(3, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (3,)).astype(np.int64)

        buffer.add_batch(states=states, actions=actions)
        assert len(buffer) == 3

    def test_capacity_reached(self) -> None:
        """Test behavior when capacity is reached."""
        buffer = Buffer(3)
        states = np.random.randn(2, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (2,)).astype(np.int64)

        # Add first batch
        buffer.add_batch(states=states, actions=actions)
        assert len(buffer) == 2

        # Add second batch to reach capacity
        buffer.add_batch(states=states, actions=actions)
        assert len(buffer) == 3  # Should be at capacity

        # Add third batch to trigger wrap-around
        buffer.add_batch(states=states, actions=actions)
        assert len(buffer) == 3  # Should still be at capacity

    def test_different_data_types(self) -> None:
        """Test buffer with different data types."""
        buffer = Buffer(10)
        states = np.random.randn(3, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (3,)).astype(np.int64)
        rewards = np.random.randn(3).astype(np.float64)
        dones = np.array([False, True, False])

        result = buffer.add_batch(states=states, actions=actions, rewards=rewards, dones=dones)

        assert len(result) == 3
        assert buffer._size == 3
        assert "states" in buffer._data
        assert "actions" in buffer._data
        assert "rewards" in buffer._data
        assert "dones" in buffer._data

    def test_sample_returns_tensors(self) -> None:
        """Test that sample returns torch tensors."""
        buffer = Buffer(10)
        states = np.random.randn(3, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (3,)).astype(np.int64)

        buffer.add_batch(states=states, actions=actions)
        sample = buffer.sample(2)

        for key, value in sample.items():
            assert isinstance(value, torch.Tensor)
            assert value.shape[0] == 2

    def test_sample_by_idxs_returns_tensors(self) -> None:
        """Test that sample_by_idxs returns torch tensors."""
        buffer = Buffer(10)
        states = np.random.randn(3, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (3,)).astype(np.int64)

        buffer.add_batch(states=states, actions=actions)
        indices = np.array([0, 1])
        sample = buffer.sample_by_idxs(indices)

        for key, value in sample.items():
            assert isinstance(value, torch.Tensor)
            assert value.shape[0] == 2


class TestReplayBuffer:
    """Test cases for the ReplayBuffer class."""

    def test_init(self) -> None:
        """Test ReplayBuffer initialization."""
        buffer = ReplayBuffer(100, (4,), np.float32, np.int64)

        assert buffer._capacity == 100
        assert buffer._ptr == 0
        assert buffer._size == 0
        assert buffer._state_type == np.float32

    def test_add_batch(self) -> None:
        """Test adding batch to ReplayBuffer."""
        buffer = ReplayBuffer(10, (4,), np.float32, np.int64)
        states = np.random.randn(3, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (3,)).astype(np.int64)
        rewards = np.random.randn(3).astype(np.float32)
        next_states = np.random.randn(3, 4).astype(np.float32)
        dones = np.array([False, True, False])

        buffer.add_batch(states, actions, rewards, next_states, dones)

        assert buffer._size == 3
        assert buffer._ptr == 3

    def test_sample(self) -> None:
        """Test sampling from ReplayBuffer."""
        buffer = ReplayBuffer(10, (4,), np.float32, np.int64)
        states = np.random.randn(3, 4).astype(np.float32)
        actions = np.random.randint(0, 2, (3,)).astype(np.int64)
        rewards = np.random.randn(3).astype(np.float32)
        next_states = np.random.randn(3, 4).astype(np.float32)
        dones = np.array([False, True, False])

        buffer.add_batch(states, actions, rewards, next_states, dones)
        experience = buffer.sample(2)

        assert isinstance(experience, Experience)
        assert experience.states.shape == (2, 4)
        assert experience.actions.shape == (2,)
        assert experience.rewards.shape == (2,)
        assert experience.next_states.shape == (2, 4)
        assert experience.dones.shape == (2,)

    def test_experience_to_device(self) -> None:
        """Test Experience.to() method."""
        states = torch.randn(2, 4)
        actions = torch.randint(0, 2, (2,))
        rewards = torch.randn(2)
        next_states = torch.randn(2, 4)
        dones = torch.tensor([False, True])

        experience = Experience(
            states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones
        )

        # Test moving to CPU (should be no-op if already on CPU)
        cpu_exp = experience.to(torch.device("cpu"))
        assert isinstance(cpu_exp, Experience)
        assert cpu_exp.states.device == torch.device("cpu")

        # Test changing dtype
        float_exp = experience.to(torch.device("cpu"), torch.float64)
        assert float_exp.states.dtype == torch.float64
        assert float_exp.actions.dtype == torch.float64  # all tensors get the same dtype
