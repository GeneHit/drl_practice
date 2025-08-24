import numpy as np
import pytest
import torch

from practice.utils_for_coding.buffer.buffer_utils import BufferNP, BufferTorch


class TestBufferNP:
    """Test cases for BufferNP class."""

    def test_init(self) -> None:
        """Test buffer initialization."""
        buffer = BufferNP(capacity=10)
        assert buffer.capacity == 10
        assert len(buffer) == 0
        assert buffer._ptr == 0
        assert buffer._size == 0
        assert not buffer._initialized

    def test_clear(self) -> None:
        """Test buffer clearing."""
        buffer = BufferNP(capacity=5)
        # Add some data first
        data = {"obs": np.array([[1, 2], [3, 4]])}
        buffer.add_batch(**data)
        assert len(buffer) == 2

        # Clear and verify
        buffer.clear()
        assert len(buffer) == 0
        assert buffer._ptr == 0
        assert buffer._size == 0

    def test_add_batch_empty(self) -> None:
        """Test adding empty batch."""
        buffer = BufferNP(capacity=5)
        indices = buffer.add_batch()
        assert len(indices) == 0
        assert len(buffer) == 0

    def test_add_batch_zero_size(self) -> None:
        """Test adding batch with zero-sized arrays."""
        buffer = BufferNP(capacity=5)
        data = {"obs": np.array([]).reshape(0, 2)}
        indices = buffer.add_batch(**data)
        assert len(indices) == 0
        assert len(buffer) == 0

    def test_add_batch_single(self) -> None:
        """Test adding a single batch without wraparound."""
        buffer = BufferNP(capacity=5)
        data = {
            "obs": np.array([[1, 2], [3, 4]]),
            "action": np.array([0, 1]),
        }
        indices = buffer.add_batch(**data)

        assert len(buffer) == 2
        assert buffer._ptr == 2
        assert np.array_equal(indices, np.array([0, 1]))
        assert np.array_equal(buffer._data["obs"][:2], data["obs"])
        assert np.array_equal(buffer._data["action"][:2], data["action"])

    def test_add_batch_multiple(self) -> None:
        """Test adding multiple batches."""
        buffer = BufferNP(capacity=5)

        # First batch
        data1 = {"obs": np.array([[1, 2]])}
        indices1 = buffer.add_batch(**data1)
        assert np.array_equal(indices1, np.array([0]))
        assert len(buffer) == 1

        # Second batch
        data2 = {"obs": np.array([[3, 4], [5, 6]])}
        indices2 = buffer.add_batch(**data2)
        assert np.array_equal(indices2, np.array([1, 2]))
        assert len(buffer) == 3

    def test_add_batch_wraparound(self) -> None:
        """Test adding batch with wraparound."""
        buffer = BufferNP(capacity=3)

        # Fill buffer
        data1 = {"obs": np.array([[1], [2], [3]])}
        buffer.add_batch(**data1)
        assert len(buffer) == 3
        assert buffer._ptr == 0  # Wrapped around

        # Add more data causing wraparound
        data2 = {"obs": np.array([[4], [5]])}
        indices = buffer.add_batch(**data2)
        expected_indices = np.array([0, 1])
        assert np.array_equal(indices, expected_indices)
        assert len(buffer) == 3  # Still at capacity
        assert buffer._ptr == 2

    def test_sample_insufficient_data(self) -> None:
        """Test sampling when buffer doesn't have enough data."""
        buffer = BufferNP(capacity=5)
        data = {"obs": np.array([[1, 2]])}
        buffer.add_batch(**data)

        with pytest.raises(AssertionError, match="Buffer size 1, but required 2"):
            buffer.sample(2)

    def test_sample_valid(self) -> None:
        """Test valid sampling."""
        buffer = BufferNP(capacity=5)
        data = {
            "obs": np.array([[1, 2], [3, 4], [5, 6]]),
            "action": np.array([0, 1, 2]),
        }
        buffer.add_batch(**data)

        sampled = buffer.sample(2)
        assert "obs" in sampled
        assert "action" in sampled
        assert sampled["obs"].shape == (2, 2)
        assert sampled["action"].shape == (2,)

    def test_sample_by_idxs_invalid_indices(self) -> None:
        """Test sampling with invalid indices."""
        buffer = BufferNP(capacity=5)
        data = {"obs": np.array([[1, 2], [3, 4]])}
        buffer.add_batch(**data)

        # Test negative indices
        with pytest.raises(ValueError, match="Invalid indices"):
            buffer.sample_by_idxs(np.array([-1, 0]))

        # Test out of range indices
        with pytest.raises(ValueError, match="Invalid indices"):
            buffer.sample_by_idxs(np.array([0, 2]))

    def test_sample_by_idxs_valid(self) -> None:
        """Test valid sampling by indices."""
        buffer = BufferNP(capacity=5)
        data = {
            "obs": np.array([[1, 2], [3, 4], [5, 6]]),
            "action": np.array([10, 20, 30]),
        }
        buffer.add_batch(**data)

        indices = np.array([0, 2])
        sampled = buffer.sample_by_idxs(indices)

        expected_obs = np.array([[1, 2], [5, 6]])
        expected_action = np.array([10, 30])

        assert np.array_equal(sampled["obs"], expected_obs)
        assert np.array_equal(sampled["action"], expected_action)

    def test_dataloader_not_initialized(self) -> None:
        """Test dataloader on uninitialized buffer."""
        buffer = BufferNP(capacity=5)
        with pytest.raises(AssertionError, match="Memory not initialized"):
            list(buffer.dataloader(2))

    def test_dataloader_no_shuffle(self) -> None:
        """Test dataloader without shuffling."""
        buffer = BufferNP(capacity=10)
        data = {"obs": np.array([[i] for i in range(5)])}
        buffer.add_batch(**data)

        batches = list(buffer.dataloader(batch_size=2, shuffle=False))
        assert len(batches) == 3  # 5 samples, batch_size=2 -> 3 batches

        # First batch should have indices [0, 1]
        assert np.array_equal(batches[0]["obs"], np.array([[0], [1]]))
        # Second batch should have indices [2, 3]
        assert np.array_equal(batches[1]["obs"], np.array([[2], [3]]))
        # Third batch should have index [4]
        assert np.array_equal(batches[2]["obs"], np.array([[4]]))

    def test_dataloader_with_shuffle(self) -> None:
        """Test dataloader with shuffling."""
        buffer = BufferNP(capacity=10)
        data = {"obs": np.array([[i] for i in range(5)])}
        buffer.add_batch(**data)

        # Set seed for reproducible test
        np.random.seed(42)
        batches = list(buffer.dataloader(batch_size=2, shuffle=True))
        assert len(batches) == 3

        # Collect all sampled data
        all_sampled = np.concatenate([batch["obs"] for batch in batches])
        all_original = np.array([[i] for i in range(5)])

        # Should contain same data but potentially different order
        assert len(all_sampled) == len(all_original)
        for item in all_original:
            assert any(np.array_equal(item, sampled_item) for sampled_item in all_sampled)


class TestBufferTorch:
    """Test cases for BufferTorch class."""

    def test_init(self) -> None:
        """Test buffer initialization."""
        buffer = BufferTorch(capacity=10)
        assert buffer.capacity == 10
        assert len(buffer) == 0
        assert buffer._ptr == 0
        assert buffer._size == 0
        assert not buffer._initialized

    def test_clear(self) -> None:
        """Test buffer clearing."""
        buffer = BufferTorch(capacity=5)
        # Add some data first
        data = {"obs": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        buffer.add_batch(**data)
        assert len(buffer) == 2

        # Clear and verify
        buffer.clear()
        assert len(buffer) == 0
        assert buffer._ptr == 0
        assert buffer._size == 0

    def test_add_batch_empty(self) -> None:
        """Test adding empty batch."""
        buffer = BufferTorch(capacity=5)
        indices = buffer.add_batch()
        assert len(indices) == 0
        assert len(buffer) == 0

    def test_add_batch_zero_size(self) -> None:
        """Test adding batch with zero-sized tensors."""
        buffer = BufferTorch(capacity=5)
        data = {"obs": torch.empty(0, 2)}
        indices = buffer.add_batch(**data)
        assert len(indices) == 0
        assert len(buffer) == 0

    def test_add_batch_single(self) -> None:
        """Test adding a single batch without wraparound."""
        buffer = BufferTorch(capacity=5)
        data = {
            "obs": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "action": torch.tensor([0, 1]),
        }
        indices = buffer.add_batch(**data)

        assert len(buffer) == 2
        assert buffer._ptr == 2
        assert torch.equal(indices, torch.tensor([0, 1]))
        assert torch.equal(buffer._data["obs"][:2], data["obs"])
        assert torch.equal(buffer._data["action"][:2], data["action"])

    def test_add_batch_multiple(self) -> None:
        """Test adding multiple batches."""
        buffer = BufferTorch(capacity=5)

        # First batch
        data1 = {"obs": torch.tensor([[1.0, 2.0]])}
        indices1 = buffer.add_batch(**data1)
        assert torch.equal(indices1, torch.tensor([0]))
        assert len(buffer) == 1

        # Second batch
        data2 = {"obs": torch.tensor([[3.0, 4.0], [5.0, 6.0]])}
        indices2 = buffer.add_batch(**data2)
        assert torch.equal(indices2, torch.tensor([1, 2]))
        assert len(buffer) == 3

    def test_add_batch_wraparound(self) -> None:
        """Test adding batch with wraparound."""
        buffer = BufferTorch(capacity=3)

        # Fill buffer
        data1 = {"obs": torch.tensor([[1.0], [2.0], [3.0]])}
        buffer.add_batch(**data1)
        assert len(buffer) == 3
        assert buffer._ptr == 0  # Wrapped around

        # Add more data causing wraparound
        data2 = {"obs": torch.tensor([[4.0], [5.0]])}
        indices = buffer.add_batch(**data2)
        expected_indices = torch.tensor([0, 1])
        assert torch.equal(indices, expected_indices)
        assert len(buffer) == 3  # Still at capacity
        assert buffer._ptr == 2

    def test_sample_insufficient_data(self) -> None:
        """Test sampling when buffer doesn't have enough data."""
        buffer = BufferTorch(capacity=5)
        data = {"obs": torch.tensor([[1.0, 2.0]])}
        buffer.add_batch(**data)

        with pytest.raises(AssertionError, match="Buffer size 1, but required 2"):
            buffer.sample(2)

    def test_sample_valid(self) -> None:
        """Test valid sampling."""
        buffer = BufferTorch(capacity=5)
        data = {
            "obs": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "action": torch.tensor([0, 1, 2]),
        }
        buffer.add_batch(**data)

        sampled = buffer.sample(2)
        assert "obs" in sampled
        assert "action" in sampled
        assert sampled["obs"].shape == (2, 2)
        assert sampled["action"].shape == (2,)

    def test_sample_by_idxs_invalid_indices(self) -> None:
        """Test sampling with invalid indices."""
        buffer = BufferTorch(capacity=5)
        data = {"obs": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        buffer.add_batch(**data)

        # Test negative indices
        with pytest.raises(ValueError, match="Invalid indices"):
            buffer.sample_by_idxs(torch.tensor([-1, 0]))

        # Test out of range indices
        with pytest.raises(ValueError, match="Invalid indices"):
            buffer.sample_by_idxs(torch.tensor([0, 2]))

    def test_sample_by_idxs_valid(self) -> None:
        """Test valid sampling by indices."""
        buffer = BufferTorch(capacity=5)
        data = {
            "obs": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "action": torch.tensor([10, 20, 30]),
        }
        buffer.add_batch(**data)

        indices = torch.tensor([0, 2])
        sampled = buffer.sample_by_idxs(indices)

        expected_obs = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        expected_action = torch.tensor([10, 30])

        assert torch.equal(sampled["obs"], expected_obs)
        assert torch.equal(sampled["action"], expected_action)

    def test_dataloader_not_initialized(self) -> None:
        """Test dataloader on uninitialized buffer."""
        buffer = BufferTorch(capacity=5)
        with pytest.raises(AssertionError, match="Memory not initialized"):
            list(buffer.dataloader(2))

    def test_dataloader_no_shuffle(self) -> None:
        """Test dataloader without shuffling."""
        buffer = BufferTorch(capacity=10)
        data = {"obs": torch.tensor([[float(i)] for i in range(5)])}
        buffer.add_batch(**data)

        batches = list(buffer.dataloader(batch_size=2, shuffle=False))
        assert len(batches) == 3  # 5 samples, batch_size=2 -> 3 batches

        # First batch should have indices [0, 1]
        assert torch.equal(batches[0]["obs"], torch.tensor([[0.0], [1.0]]))
        # Second batch should have indices [2, 3]
        assert torch.equal(batches[1]["obs"], torch.tensor([[2.0], [3.0]]))
        # Third batch should have index [4]
        assert torch.equal(batches[2]["obs"], torch.tensor([[4.0]]))

    def test_dataloader_with_shuffle(self) -> None:
        """Test dataloader with shuffling."""
        buffer = BufferTorch(capacity=10)
        data = {"obs": torch.tensor([[float(i)] for i in range(5)])}
        buffer.add_batch(**data)

        # Set seed for reproducible test
        torch.manual_seed(42)
        batches = list(buffer.dataloader(batch_size=2, shuffle=True))
        assert len(batches) == 3

        # Collect all sampled data
        all_sampled = torch.cat([batch["obs"] for batch in batches])
        all_original = torch.tensor([[float(i)] for i in range(5)])

        # Should contain same data but potentially different order
        assert len(all_sampled) == len(all_original)
        for item in all_original:
            assert any(torch.equal(item, sampled_item) for sampled_item in all_sampled)

    def test_device_consistency(self) -> None:
        """Test that buffer maintains device consistency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda:0")
        buffer = BufferTorch(capacity=5)

        data = {
            "obs": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device),
            "action": torch.tensor([0, 1], device=device),
        }
        buffer.add_batch(**data)

        # Sample and verify device
        sampled = buffer.sample(2)
        assert sampled["obs"].device == device
        assert sampled["action"].device == device

        # Sample by indices and verify device
        indices = torch.tensor([0, 1], device=device)
        sampled_by_idx = buffer.sample_by_idxs(indices)
        assert sampled_by_idx["obs"].device == device
        assert sampled_by_idx["action"].device == device


class TestBufferComparison:
    """Test cases comparing BufferNP and BufferTorch behavior."""

    def test_equivalent_behavior(self) -> None:
        """Test that both buffers behave equivalently for the same operations."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Create buffers
        np_buffer = BufferNP(capacity=5)
        torch_buffer = BufferTorch(capacity=5)

        # Add equivalent data
        np_data = {"obs": np.array([[1, 2], [3, 4], [5, 6]])}
        torch_data = {"obs": torch.tensor([[1, 2], [3, 4], [5, 6]])}

        np_indices = np_buffer.add_batch(**np_data)
        torch_indices = torch_buffer.add_batch(**torch_data)

        # Verify equivalent indices
        assert np.array_equal(np_indices, torch_indices.numpy())

        # Verify equivalent sizes
        assert len(np_buffer) == len(torch_buffer)
        assert np_buffer.capacity == torch_buffer.capacity

    def test_wraparound_equivalence(self) -> None:
        """Test that wraparound behavior is equivalent."""
        np_buffer = BufferNP(capacity=3)
        torch_buffer = BufferTorch(capacity=3)

        # Fill buffers
        np_data1 = {"obs": np.array([[1], [2], [3]])}
        torch_data1 = {"obs": torch.tensor([[1], [2], [3]])}

        np_buffer.add_batch(**np_data1)
        torch_buffer.add_batch(**torch_data1)

        # Add wraparound data
        np_data2 = {"obs": np.array([[4], [5]])}
        torch_data2 = {"obs": torch.tensor([[4], [5]])}

        np_indices = np_buffer.add_batch(**np_data2)
        torch_indices = torch_buffer.add_batch(**torch_data2)

        # Verify equivalent wraparound indices
        assert np.array_equal(np_indices, torch_indices.numpy())
        assert len(np_buffer) == len(torch_buffer)
