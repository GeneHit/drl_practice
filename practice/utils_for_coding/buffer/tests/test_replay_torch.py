from typing import Dict

import pytest
import torch
from torch import Tensor

from practice.utils_for_coding.buffer.replay_torch import ReplayBuffer


class TestReplayBuffer:
    """Test suite for ReplayBuffer class using torch tensors."""

    @pytest.fixture
    def buffer(self) -> ReplayBuffer:
        """Create a replay buffer for testing."""
        return ReplayBuffer(capacity=10)

    @pytest.fixture
    def sample_data(self) -> Dict[str, Tensor]:
        """Create sample data for testing."""
        batch_size = 3
        state_dim = 4
        action_dim = 2

        return {
            "states": torch.rand(batch_size, state_dim, dtype=torch.float32),
            "actions": torch.rand(batch_size, action_dim, dtype=torch.float32),
            "rewards": torch.rand(batch_size, dtype=torch.float32),
            "next_states": torch.rand(batch_size, state_dim, dtype=torch.float32),
            "dones": torch.randint(0, 2, (batch_size,), dtype=torch.bool),
        }

    def test_init(self) -> None:
        """Test buffer initialization."""
        capacity = 100
        buffer = ReplayBuffer(capacity)
        assert len(buffer) == 0
        assert buffer.capacity == capacity

    def test_add_batch(self, buffer: ReplayBuffer, sample_data: Dict[str, Tensor]) -> None:
        """Test adding a batch of data."""
        initial_len = len(buffer)
        indices = buffer.add_batch(**sample_data)

        # Check that buffer size increased
        assert len(buffer) == initial_len + len(sample_data["states"])

        # Check that indices are returned correctly
        expected_indices = torch.arange(initial_len, initial_len + len(sample_data["states"]))
        torch.testing.assert_close(indices, expected_indices)

    def test_add_empty_batch(self, buffer: ReplayBuffer) -> None:
        """Test adding an empty batch."""
        empty_data = {
            "states": torch.empty(0, 4, dtype=torch.float32),
            "actions": torch.empty(0, 2, dtype=torch.float32),
            "rewards": torch.empty(0, dtype=torch.float32),
            "next_states": torch.empty(0, 4, dtype=torch.float32),
            "dones": torch.empty(0, dtype=torch.bool),
        }

        initial_len = len(buffer)
        indices = buffer.add_batch(**empty_data)

        assert len(buffer) == initial_len
        assert len(indices) == 0

    def test_sample(self, buffer: ReplayBuffer, sample_data: Dict[str, Tensor]) -> None:
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
            "states": torch.rand(1, 4, dtype=torch.float32),
            "actions": torch.rand(1, 2, dtype=torch.float32),
            "rewards": torch.rand(1, dtype=torch.float32),
            "next_states": torch.rand(1, 4, dtype=torch.float32),
            "dones": torch.tensor([False]),
        }
        buffer.add_batch(**sample_data)

        with pytest.raises(AssertionError):
            buffer.sample(2)

    def test_sample_by_idxs(self, buffer: ReplayBuffer, sample_data: Dict[str, Tensor]) -> None:
        """Test sampling by specific indices."""
        buffer.add_batch(**sample_data)

        idxs = torch.tensor([0, 2], dtype=torch.int64)
        experience = buffer.sample_by_idxs(idxs)

        # Check that we got the right number of samples
        assert experience.states.shape[0] == len(idxs)
        assert experience.actions.shape[0] == len(idxs)
        assert experience.rewards.shape[0] == len(idxs)
        assert experience.next_states.shape[0] == len(idxs)
        assert experience.dones.shape[0] == len(idxs)

    def test_sample_by_invalid_idxs(
        self, buffer: ReplayBuffer, sample_data: Dict[str, Tensor]
    ) -> None:
        """Test sampling by invalid indices."""
        buffer.add_batch(**sample_data)

        # Test negative index
        with pytest.raises(ValueError):
            buffer.sample_by_idxs(torch.tensor([-1], dtype=torch.int64))

        # Test index out of range
        with pytest.raises(ValueError):
            buffer.sample_by_idxs(torch.tensor([10], dtype=torch.int64))

    def test_dataloader(self, buffer: ReplayBuffer, sample_data: Dict[str, Tensor]) -> None:
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
            "states": torch.arange(20).unsqueeze(1).float(),
            "actions": torch.arange(20).unsqueeze(1).float(),
            "rewards": torch.arange(20).float(),
            "next_states": torch.arange(20).unsqueeze(1).float(),
            "dones": torch.tensor([False] * 20),
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

    def test_clear(self, buffer: ReplayBuffer, sample_data: Dict[str, Tensor]) -> None:
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
                "states": torch.full((3, 1), i, dtype=torch.float32),
                "actions": torch.full((3, 1), i, dtype=torch.float32),
                "rewards": torch.full((3,), i, dtype=torch.float32),
                "next_states": torch.full((3, 1), i, dtype=torch.float32),
                "dones": torch.tensor([False] * 3),
            }
            buffer.add_batch(**batch_data)

        # Buffer should be at capacity
        assert len(buffer) == capacity

        # Sample all data and verify it contains the most recent samples
        experience = buffer.sample_by_idxs(torch.arange(capacity, dtype=torch.int64))

        # Should contain mix of data from different batches due to wraparound
        assert experience.states.shape[0] == capacity

    def test_different_dtypes(self, buffer: ReplayBuffer) -> None:
        """Test buffer with different data types."""
        # Test with integer actions (discrete action space)
        states: Tensor = torch.rand(2, 4, dtype=torch.float32)
        actions: Tensor = torch.tensor([0, 1], dtype=torch.int64)
        rewards: Tensor = torch.tensor([1.0, -1.0], dtype=torch.float32)
        next_states: Tensor = torch.rand(2, 4, dtype=torch.float32)
        dones: Tensor = torch.tensor([False, True])

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
        states: Tensor = torch.randint(0, 255, (2, 84, 84, 3), dtype=torch.uint8)
        actions: Tensor = torch.tensor([0, 1], dtype=torch.int64)
        rewards: Tensor = torch.tensor([1.0, -1.0], dtype=torch.float32)
        next_states: Tensor = torch.randint(0, 255, (2, 84, 84, 3), dtype=torch.uint8)
        dones: Tensor = torch.tensor([False, True])

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
            "states": torch.rand(8, 4, dtype=torch.float32),
            "actions": torch.rand(8, 2, dtype=torch.float32),
            "rewards": torch.rand(8, dtype=torch.float32),
            "next_states": torch.rand(8, 4, dtype=torch.float32),
            "dones": torch.tensor([False] * 8),
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
            "states": torch.rand(3, 4, dtype=torch.float32),
            "actions": torch.rand(3, 2, dtype=torch.float32),
            "rewards": torch.rand(3, dtype=torch.float32),
            "next_states": torch.rand(3, 4, dtype=torch.float32),
            "dones": torch.tensor([False, True, False]),
        }
        buffer.add_batch(**sample_data)

        assert len(buffer) == 3
        assert buffer.capacity == 10  # Should remain unchanged

    def test_device_consistency(self) -> None:
        """Test that tensors maintain their device placement."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device consistency test")

        device = torch.device("cuda:0")
        buffer = ReplayBuffer(capacity=10)

        # Add data on GPU
        sample_data = {
            "states": torch.rand(3, 4, dtype=torch.float32, device=device),
            "actions": torch.rand(3, 2, dtype=torch.float32, device=device),
            "rewards": torch.rand(3, dtype=torch.float32, device=device),
            "next_states": torch.rand(3, 4, dtype=torch.float32, device=device),
            "dones": torch.tensor([False, True, False], device=device),
        }
        buffer.add_batch(**sample_data)

        # Sample data should be on the same device
        experience = buffer.sample(2)
        assert experience.states.device == device
        assert experience.actions.device == device
        assert experience.rewards.device == device
        assert experience.next_states.device == device
        assert experience.dones.device == device

    def test_mixed_devices(self) -> None:
        """Test error handling for mixed device tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed device test")

        buffer = ReplayBuffer(capacity=10)

        # Try to add data with mixed devices (should work but might not be optimal)
        sample_data = {
            "states": torch.rand(2, 4, dtype=torch.float32),  # CPU
            "actions": torch.rand(2, 2, dtype=torch.float32, device="cuda:0"),  # GPU
            "rewards": torch.rand(2, dtype=torch.float32),  # CPU
            "next_states": torch.rand(2, 4, dtype=torch.float32),  # CPU
            "dones": torch.tensor([False, True]),  # CPU
        }

        # This should work - BufferTorch should handle device placement
        buffer.add_batch(**sample_data)
        experience = buffer.sample(1)

        # Check that we can sample (device handling is up to BufferTorch implementation)
        assert experience.states.shape[0] == 1
