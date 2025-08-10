import numpy as np
import pytest

from practice.exercise2_dqn.e24_rainbow.sum_tree import SumTree


class TestSumTree:
    """Test cases for the SumTree class."""

    def test_init_valid_capacity(self) -> None:
        """Test SumTree initialization with valid capacity."""
        capacity = 100
        tree = SumTree(capacity)

        assert tree._capacity == capacity
        assert len(tree._tree) == 2 * capacity - 1
        assert len(tree._used) == capacity
        assert tree._leaf_start == capacity - 1
        assert len(tree) == 0  # No items added yet
        assert tree.total_priority == 0.0

    @pytest.mark.parametrize("capacity", [0, -1, 1])
    def test_init_invalid_capacity(self, capacity: int) -> None:
        """Test SumTree initialization with invalid capacity."""
        with pytest.raises(ValueError, match="capacity must be greater than 1"):
            SumTree(capacity)

    def test_update_single_priority(self) -> None:
        """Test updating a single priority."""
        tree = SumTree(10)
        idx = 5
        priority = 2.5

        tree.update(np.array([idx]), np.array([priority]))

        assert tree._tree[tree._leaf_start + idx] == priority
        assert tree._used[idx]
        assert tree.total_priority == priority
        assert len(tree) == 1

    def test_update_multiple_priorities(self) -> None:
        """Test updating multiple priorities."""
        tree = SumTree(10)
        idxs = np.array([0, 2, 4])
        priorities = np.array([1.0, 2.0, 3.0])

        tree.update(idxs, priorities)

        for idx, priority in zip(idxs, priorities):
            assert tree._tree[tree._leaf_start + idx] == priority
            assert tree._used[idx]
        assert tree.total_priority == 6.0
        assert len(tree) == 3

    def test_update_overwrite_existing(self) -> None:
        """Test updating existing priorities."""
        tree = SumTree(10)
        idx = 5
        old_priority = 2.0
        new_priority = 4.0

        # First update
        tree.update(np.array([idx]), np.array([old_priority]))
        assert tree.total_priority == old_priority

        # Second update (overwrite)
        tree.update(np.array([idx]), np.array([new_priority]))
        assert tree._tree[tree._leaf_start + idx] == new_priority
        assert tree.total_priority == new_priority
        assert len(tree) == 1  # Still only one item

    def test_update_small_delta_optimization(self) -> None:
        """Test that very small priority changes are ignored for optimization."""
        tree = SumTree(10)
        idx = 0
        priority = 1.0

        # Set initial priority
        tree.update(np.array([idx]), np.array([priority]))
        initial_total = tree.total_priority

        # Update with very small change (should be ignored due to delta < 1e-6)
        small_change = priority + 1e-7
        tree.update(np.array([idx]), np.array([small_change]))

        # Should remain unchanged due to optimization
        assert tree.total_priority == initial_total

    def test_update_validation_errors(self) -> None:
        """Test update method validation errors."""
        tree = SumTree(10)

        # Test mismatched array dimensions
        with pytest.raises(ValueError, match="idxs/priorities must be 1-D and of equal length"):
            tree.update(np.array([0]), np.array([[1.0, 2.0]]))

        # Test mismatched array lengths
        with pytest.raises(ValueError, match="idxs/priorities must be 1-D and of equal length"):
            tree.update(np.array([0, 1]), np.array([1.0]))

        # Test out of range indices
        with pytest.raises(ValueError, match="index out of range"):
            tree.update(np.array([-1]), np.array([1.0]))

        with pytest.raises(ValueError, match="index out of range"):
            tree.update(np.array([10]), np.array([1.0]))

        # Test non-finite priorities
        with pytest.raises(ValueError, match="priorities must be finite"):
            tree.update(np.array([0]), np.array([np.nan]))

        with pytest.raises(ValueError, match="priorities must be finite"):
            tree.update(np.array([0]), np.array([np.inf]))

        # Test negative priorities
        with pytest.raises(ValueError, match="priorities must be non-negative"):
            tree.update(np.array([0]), np.array([-1.0]))

    def test_update_negative_priority_cleanup(self) -> None:
        """Test that slightly negative priorities are cleaned up to 0."""
        tree = SumTree(10)
        idx = 0

        # Very small negative value should be cleaned up to 0
        tree.update(np.array([idx]), np.array([-1e-10]))
        assert tree._tree[tree._leaf_start + idx] == 0.0
        assert tree.total_priority == 0.0

    def test_sample_empty_tree(self) -> None:
        """Test sampling from empty tree."""
        tree = SumTree(10)

        with pytest.raises(ValueError, match="buffer size=0, but requested 1 samples"):
            tree.sample(1)

    def test_sample_single_item(self) -> None:
        """Test sampling a single item."""
        tree = SumTree(10)
        idx = 5
        priority = 2.0

        tree.update(np.array([idx]), np.array([priority]))
        data_idxs, priorities = tree.sample(1)

        assert len(data_idxs) == 1
        assert len(priorities) == 1
        assert data_idxs[0] == idx
        assert priorities[0] == priority

    def test_sample_multiple_items(self) -> None:
        """Test sampling multiple items."""
        tree = SumTree(10)
        idxs = np.array([0, 2, 4])
        priorities = np.array([1.0, 2.0, 3.0])

        tree.update(idxs, priorities)
        sampled_idxs, sampled_priorities = tree.sample(2)

        assert len(sampled_idxs) == 2
        assert len(sampled_priorities) == 2
        # All sampled indices should be in the original set
        assert all(idx in idxs for idx in sampled_idxs)

    def test_sample_all_items(self) -> None:
        """Test sampling all available items."""
        tree = SumTree(5)
        idxs = np.array([0, 1, 2])
        priorities = np.array([1.0, 2.0, 3.0])

        tree.update(idxs, priorities)
        sampled_idxs, sampled_priorities = tree.sample(3)

        assert len(sampled_idxs) == 3
        assert len(sampled_priorities) == 3
        # All sampled indices should be valid (from the available set)
        assert all(idx in idxs for idx in sampled_idxs)

    def test_sample_more_than_available(self) -> None:
        """Test sampling more items than available."""
        tree = SumTree(10)
        tree.update(np.array([0, 1]), np.array([1.0, 2.0]))

        with pytest.raises(ValueError, match="buffer size=2, but requested 3 samples"):
            tree.sample(3)

    def test_sample_zero_total_priority(self) -> None:
        """Test sampling when total priority is zero."""
        tree = SumTree(10)
        tree.update(np.array([0, 1]), np.array([0.0, 0.0]))

        # Should raise error for zero total priority
        with pytest.raises(ValueError, match="total priority must be positive and finite"):
            tree.sample(1)

    def test_sample_invalid_total_priority(self) -> None:
        """Test sampling when total priority is invalid."""
        tree = SumTree(10)
        tree.update(np.array([0]), np.array([1.0]))

        # Manually corrupt the tree to test error handling
        tree._tree[0] = np.nan
        with pytest.raises(ValueError, match="total priority must be positive and finite"):
            tree.sample(1)

        tree._tree[0] = -1.0
        with pytest.raises(ValueError, match="total priority must be positive and finite"):
            tree.sample(1)

    def test_tree_structure_correctness(self) -> None:
        """Test that the tree structure maintains correct parent-child relationships."""
        capacity = 4
        tree = SumTree(capacity)

        # Fill the buffer
        indices = np.arange(capacity)
        priorities = np.array([1.0, 2.0, 3.0, 4.0])
        tree.update(indices, priorities)

        # Verify tree structure
        # Tree should have 2*4-1 = 7 nodes
        assert len(tree._tree) == 7
        assert tree._leaf_start == 3

        # Root should be sum of all priorities
        assert tree._tree[0] == 10.0  # 1 + 2 + 3 + 4

        # Leaf nodes should contain the priorities
        assert tree._tree[3] == 1.0  # data index 0
        assert tree._tree[4] == 2.0  # data index 1
        assert tree._tree[5] == 3.0  # data index 2
        assert tree._tree[6] == 4.0  # data index 3

    def test_tree_traversal_correctness(self) -> None:
        """Test that tree traversal returns correct indices and priorities."""
        tree = SumTree(4)
        indices = np.arange(4)
        priorities = np.array([1.0, 2.0, 3.0, 4.0])
        tree.update(indices, priorities)

        # Sample and verify
        sampled_idxs, sampled_priorities = tree.sample(2)

        # Check that returned priorities match the expected values
        for i, idx in enumerate(sampled_idxs):
            expected_priority = priorities[idx]
            actual_priority = sampled_priorities[i]
            assert actual_priority == expected_priority

    @pytest.mark.parametrize("capacity", [2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64])
    def test_various_capacities(self, capacity: int) -> None:
        """Test that SumTree works correctly with various capacities."""
        tree = SumTree(capacity)

        # Fill partially
        num_items = min(capacity, 10)
        indices = np.arange(num_items)
        priorities = np.random.uniform(0.1, 10.0, num_items).astype(np.float32)

        tree.update(indices, priorities)

        # Test sampling
        batch_size = min(3, len(tree))
        if batch_size > 0:
            sampled_idxs, sampled_priorities = tree.sample(batch_size)

            # Verify all indices are valid
            assert np.all(sampled_idxs >= 0)
            assert np.all(sampled_idxs < capacity)
            assert len(sampled_idxs) == batch_size
            assert len(sampled_priorities) == batch_size

    def test_stratified_sampling_distribution(self) -> None:
        """Test that stratified sampling produces reasonable distribution."""
        tree = SumTree(100)

        # Create priorities with known distribution
        indices = np.arange(10)
        priorities = np.array([1.0] * 5 + [10.0] * 5)  # Half low, half high priority
        tree.update(indices, priorities)

        # Sample many times and check distribution
        high_priority_count = 0
        total_samples = 1000

        for _ in range(total_samples):
            sampled_idxs, _ = tree.sample(1)
            if sampled_idxs[0] >= 5:  # High priority indices
                high_priority_count += 1

        # High priority items should be sampled more frequently
        # With priorities [1,1,1,1,1,10,10,10,10,10], total = 55
        # High priority items have 50/55 ≈ 0.91 probability
        high_priority_ratio = high_priority_count / total_samples
        assert high_priority_ratio > 0.7  # Should be significantly biased towards high priority

    def test_numerical_stability_large_priorities(self) -> None:
        """Test numerical stability with large priorities."""
        tree = SumTree(10)

        # Add normal priorities first
        indices = np.array([0, 1, 2])
        priorities = np.array([1.0, 2.0, 3.0])
        tree.update(indices, priorities)

        # Update with large priorities
        large_priorities = np.array([1e6, 1e7, 1e8], dtype=np.float32)
        tree.update(indices, large_priorities)

        # Should still work without overflow
        sampled_idxs, sampled_priorities = tree.sample(2)
        assert len(sampled_idxs) == 2
        assert len(sampled_priorities) == 2
        assert all(idx in indices for idx in sampled_idxs)

    def test_tree_corruption_detection(self) -> None:
        """Test that tree corruption is detected during updates."""
        tree = SumTree(10)

        # This should trigger the tree corruption detection
        # by creating a scenario where parent values become invalid
        tree.update(np.array([0]), np.array([1.0]))

        # Manually corrupt a parent node to test detection
        tree._tree[0] = np.nan

        # Next update should detect corruption
        with pytest.raises(ValueError, match="The tree get invalid values"):
            tree.update(np.array([1]), np.array([1.0]))

    def test_priority_precision_handling(self) -> None:
        """Test handling of very small priority differences."""
        tree = SumTree(10)

        # Test with small but not too small priorities (larger than 1e-6 threshold)
        small_priorities = np.array([1e-5, 1e-4, 1e-3], dtype=np.float32)
        indices = np.array([0, 1, 2])
        tree.update(indices, small_priorities)

        # Should still work
        assert len(tree) == 3
        assert tree.total_priority > 0

        sampled_idxs, sampled_priorities = tree.sample(1)
        assert len(sampled_idxs) == 1
        assert sampled_priorities[0] > 0

    def test_retrieve_method_edge_cases(self) -> None:
        """Test the _retrieve method with edge cases."""
        tree = SumTree(4)

        # Set up a specific tree structure
        priorities = np.array([0.1, 0.1, 0.1, 10.0])  # Last item has very high priority
        indices = np.arange(4)
        tree.update(indices, priorities)

        # Sample multiple times to test different traversal paths
        for _ in range(20):
            sampled_idxs, sampled_priorities = tree.sample(1)

            # All samples should be valid
            assert 0 <= sampled_idxs[0] < 4
            assert sampled_priorities[0] > 0

    def test_concurrent_operations_simulation(self) -> None:
        """Test rapid sequence of updates and samples to simulate concurrent access."""
        tree = SumTree(1000)

        # Initial setup
        initial_size = 100
        indices = np.arange(initial_size)
        priorities = np.random.uniform(0.1, 10.0, initial_size).astype(np.float32)
        tree.update(indices, priorities)

        # Rapid interleaved operations
        for round_num in range(50):
            # Sample
            if len(tree) >= 10:
                sampled_idxs, sampled_priorities = tree.sample(10)

                # Validate immediately
                assert np.all(sampled_idxs >= 0)
                assert np.all(sampled_idxs < 1000)
                assert len(sampled_idxs) == 10
                assert len(sampled_priorities) == 10

                # Update with new priorities
                new_priorities = np.random.uniform(0.1, 10.0, len(sampled_idxs)).astype(np.float32)
                tree.update(sampled_idxs, new_priorities)

    def test_memory_layout_and_indexing(self) -> None:
        """Test that memory layout and indexing work correctly."""
        capacity = 8
        tree = SumTree(capacity)

        # Verify tree structure
        assert len(tree._tree) == 2 * capacity - 1  # 15 nodes
        assert tree._leaf_start == capacity - 1  # 7
        assert len(tree._used) == capacity  # 8

        # Fill all positions
        indices = np.arange(capacity)
        priorities = np.arange(1, capacity + 1, dtype=np.float32)  # [1,2,3,4,5,6,7,8]
        tree.update(indices, priorities)

        # Check leaf positions
        for i in range(capacity):
            leaf_pos = tree._leaf_start + i
            assert tree._tree[leaf_pos] == priorities[i]

        # Check that all positions are marked as used
        assert np.all(tree._used)
        assert len(tree) == capacity

    def test_total_priority_property_consistency(self) -> None:
        """Test that total_priority property remains consistent."""
        tree = SumTree(10)

        # Initially zero
        assert tree.total_priority == 0.0

        # Add some priorities
        tree.update(np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]))
        assert tree.total_priority == 6.0

        # Update existing
        tree.update(np.array([1]), np.array([5.0]))  # Change 2.0 to 5.0
        assert tree.total_priority == 9.0  # 1.0 + 5.0 + 3.0

        # Add more
        tree.update(np.array([3, 4]), np.array([1.5, 2.5]))
        assert tree.total_priority == 13.0  # 9.0 + 1.5 + 2.5

    def test_sampling_with_zero_priorities_mixed(self) -> None:
        """Test sampling when some priorities are zero but total is positive."""
        tree = SumTree(5)

        # Mix of zero and non-zero priorities
        indices = np.array([0, 1, 2, 3])
        priorities = np.array([0.0, 1.0, 0.0, 2.0])
        tree.update(indices, priorities)

        assert tree.total_priority == 3.0
        assert len(tree) == 4

        # Should be able to sample
        sampled_idxs, sampled_priorities = tree.sample(2)
        assert len(sampled_idxs) == 2
        # Sampled items should have positive priorities
        assert all(p > 0 for p in sampled_priorities)
        # Sampled indices should be from non-zero priority items
        assert all(idx in [1, 3] for idx in sampled_idxs)

    def test_type_conversion_and_validation(self) -> None:
        """Test proper type conversion and validation."""
        tree = SumTree(10)

        # Test with different input types that should be converted
        tree.update(np.array([0, 1]), np.array([1.5, 2.5]))  # Lists should be converted to arrays
        assert len(tree) == 2
        assert tree.total_priority == 4.0

        # Test with integer priorities (should be converted to float)
        tree.update(np.array([2, 3]), np.array([3, 4], dtype=np.int32))
        assert len(tree) == 4
        assert tree.total_priority == 11.0  # 1.5 + 2.5 + 3 + 4 = 11.0

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
    def test_batch_sampling_robustness(self, batch_size: int) -> None:
        """Test batch sampling with various sizes."""
        capacity = 100
        tree = SumTree(capacity)

        # Fill with realistic data
        num_items = 50
        indices = np.arange(num_items)
        priorities = np.random.exponential(scale=1.0, size=num_items).astype(np.float32)
        priorities = np.maximum(priorities, 1e-6)  # Ensure positive

        tree.update(indices, priorities)

        # Test sampling with various batch sizes
        if len(tree) >= batch_size:
            sampled_idxs, sampled_priorities = tree.sample(batch_size)

            # Validate results
            assert len(sampled_idxs) == batch_size
            assert len(sampled_priorities) == batch_size
            assert np.all(sampled_idxs >= 0)
            assert np.all(sampled_idxs < capacity)
            assert np.all(sampled_priorities > 0)
