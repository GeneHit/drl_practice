import numpy as np
import pytest

from practice.exercise2_dqn.e24_rainbow.per_exercise import (
    NStepReplay,
    PERBuffer,
    PERBufferConfig,
    _NStepData,
    _NStepDeque,
    _ReplayBuffer,
    _StepData,
    _SumTree,
)


class TestSumTree:
    """Test cases for the _SumTree class."""

    def test_init(self) -> None:
        """Test SumTree initialization."""
        capacity = 100
        tree = _SumTree(capacity)

        assert tree._capacity == capacity
        assert len(tree._tree) == 2 * capacity - 1
        assert len(tree._used) == capacity
        assert tree._tree_height == int(np.ceil(np.log2(capacity))) + 1
        assert tree._capacity_minus_1 == capacity - 1

    def test_update_single(self) -> None:
        """Test updating a single priority."""
        tree = _SumTree(10)
        idx = 5
        priority = 2.5

        tree.update(np.array([idx]), np.array([priority]))

        assert tree._tree[idx + tree._capacity_minus_1] == priority
        assert tree._used[idx]
        assert tree.total_priority == priority

    def test_update_multiple(self) -> None:
        """Test updating multiple priorities."""
        tree = _SumTree(10)
        idxs = np.array([0, 2, 4])
        priorities = np.array([1.0, 2.0, 3.0])

        tree.update(idxs, priorities)

        for idx, priority in zip(idxs, priorities):
            assert tree._tree[idx + tree._capacity_minus_1] == priority
            assert tree._used[idx]
        assert tree.total_priority == 6.0

    def test_update_overwrite(self) -> None:
        """Test updating existing priorities."""
        tree = _SumTree(10)
        idx = 5
        old_priority = 2.0
        new_priority = 4.0

        # First update
        tree.update(np.array([idx]), np.array([old_priority]))
        assert tree.total_priority == old_priority

        # Second update
        tree.update(np.array([idx]), np.array([new_priority]))
        assert tree._tree[idx + tree._capacity_minus_1] == new_priority
        assert tree.total_priority == new_priority

    def test_sample_empty(self) -> None:
        """Test sampling from empty tree."""
        tree = _SumTree(10)

        with pytest.raises(AssertionError):
            tree.sample(1)

    def test_sample_single(self) -> None:
        """Test sampling a single item."""
        tree = _SumTree(10)
        idx = 5
        priority = 2.0

        tree.update(np.array([idx]), np.array([priority]))
        idxs, priorities = tree.sample(1)

        assert len(idxs) == 1
        assert len(priorities) == 1
        assert idxs[0] == idx
        assert priorities[0] == priority

    def test_sample_multiple(self) -> None:
        """Test sampling multiple items."""
        tree = _SumTree(10)
        idxs = np.array([0, 2, 4])
        priorities = np.array([1.0, 2.0, 3.0])

        tree.update(idxs, priorities)
        sampled_idxs, sampled_priorities = tree.sample(2)

        assert len(sampled_idxs) == 2
        assert len(sampled_priorities) == 2
        # All sampled indices should be in the original set
        assert all(idx in idxs for idx in sampled_idxs)

    def test_sample_uniform_fallback(self) -> None:
        """Test uniform sampling fallback when total priority is 0."""
        tree = _SumTree(10)
        idxs = np.array([0, 2, 4])
        priorities = np.array([0.0, 0.0, 0.0])

        tree.update(idxs, priorities)
        sampled_idxs, sampled_priorities = tree.sample(2)

        assert len(sampled_idxs) == 2
        assert len(sampled_priorities) == 2
        # Should fall back to uniform sampling
        assert all(idx in idxs for idx in sampled_idxs)

    def test_len(self) -> None:
        """Test tree length."""
        tree = _SumTree(10)
        assert len(tree) == 0

        tree.update(np.array([0, 2]), np.array([1.0, 2.0]))
        assert len(tree) == 2

    def test_total_priority(self) -> None:
        """Test total priority property."""
        tree = _SumTree(10)
        assert tree.total_priority == 0.0

        tree.update(np.array([0, 2]), np.array([1.0, 2.0]))
        assert tree.total_priority == 3.0


class TestReplayBuffer:
    """Test cases for the _ReplayBuffer class."""

    def test_init(self) -> None:
        """Test ReplayBuffer initialization."""
        capacity = 100
        buffer = _ReplayBuffer(capacity)

        assert buffer._capacity == capacity
        assert len(buffer) == 0

    def test_add_batch(self) -> None:
        """Test adding batch to buffer."""
        buffer = _ReplayBuffer(10)
        n_step_data = _NStepData(
            states=np.random.randn(3, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (3,)).astype(np.int64),
            rewards=np.random.randn(3).astype(np.float32),
            n_next_states=np.random.randn(3, 4).astype(np.float32),
            n_dones=np.array([False, True, False]),
            n=np.array([1, 2, 1], dtype=np.int16),
        )

        result = buffer.add_batch(n_step_data)

        assert len(result) == 3
        assert len(buffer) == 3

    def test_sample(self) -> None:
        """Test sampling from buffer."""
        buffer = _ReplayBuffer(10)
        n_step_data = _NStepData(
            states=np.random.randn(3, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (3,)).astype(np.int64),
            rewards=np.random.randn(3).astype(np.float32),
            n_next_states=np.random.randn(3, 4).astype(np.float32),
            n_dones=np.array([False, True, False]),
            n=np.array([1, 2, 1], dtype=np.int16),
        )

        buffer.add_batch(n_step_data)
        sampled_data = buffer.sample(2)

        assert isinstance(sampled_data, _NStepData)
        assert sampled_data.states.shape == (2, 4)
        assert sampled_data.actions.shape == (2,)
        assert sampled_data.rewards.shape == (2,)
        assert sampled_data.n_next_states.shape == (2, 4)
        assert sampled_data.n_dones.shape == (2,)
        assert sampled_data.n.shape == (2,)

    def test_sample_by_idxs(self) -> None:
        """Test sampling by specific indices."""
        buffer = _ReplayBuffer(10)
        n_step_data = _NStepData(
            states=np.random.randn(3, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (3,)).astype(np.int64),
            rewards=np.random.randn(3).astype(np.float32),
            n_next_states=np.random.randn(3, 4).astype(np.float32),
            n_dones=np.array([False, True, False]),
            n=np.array([1, 2, 1], dtype=np.int16),
        )

        buffer.add_batch(n_step_data)
        indices = [0, 2]
        sampled_data = buffer.sample_by_idxs(indices)

        assert isinstance(sampled_data, _NStepData)
        assert sampled_data.states.shape == (2, 4)
        assert sampled_data.actions.shape == (2,)
        assert sampled_data.rewards.shape == (2,)
        assert sampled_data.n_next_states.shape == (2, 4)
        assert sampled_data.n_dones.shape == (2,)
        assert sampled_data.n.shape == (2,)

    def test_len(self) -> None:
        """Test buffer length."""
        buffer = _ReplayBuffer(10)
        assert len(buffer) == 0

        n_step_data = _NStepData(
            states=np.random.randn(3, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (3,)).astype(np.int64),
            rewards=np.random.randn(3).astype(np.float32),
            n_next_states=np.random.randn(3, 4).astype(np.float32),
            n_dones=np.array([False, True, False]),
            n=np.array([1, 2, 1], dtype=np.int16),
        )

        buffer.add_batch(n_step_data)
        assert len(buffer) == 3


class TestNStepDeque:
    """Test cases for the _NStepDeque class."""

    def test_init(self) -> None:
        """Test NStepDeque initialization."""
        n_step = 3
        gamma = 0.99
        deque = _NStepDeque(n_step, gamma)

        assert deque._n_step == n_step
        assert deque._gamma == gamma
        assert not deque._init_flag

    def test_append_single_env(self) -> None:
        """Test appending data for single environment."""
        deque = _NStepDeque(n_step=3, gamma=0.99)

        # Add first step
        step_data = _StepData(
            states=np.random.randn(1, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (1,)).astype(np.int64),
            rewards=np.array([1.0], dtype=np.float32),
            next_states=np.random.randn(1, 4).astype(np.float32),
            dones=np.array([False], dtype=np.bool_),
            env_idxs=np.array([0], dtype=np.int16),
        )

        result = deque.append(step_data)
        assert result is None  # Not enough steps yet

        # Add second step
        step_data = _StepData(
            states=np.random.randn(1, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (1,)).astype(np.int64),
            rewards=np.array([2.0], dtype=np.float32),
            next_states=np.random.randn(1, 4).astype(np.float32),
            dones=np.array([False], dtype=np.bool_),
            env_idxs=np.array([0], dtype=np.int16),
        )

        result = deque.append(step_data)
        assert result is None  # Still not enough steps

        # Add third step (deque is full)
        step_data = _StepData(
            states=np.random.randn(1, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (1,)).astype(np.int64),
            rewards=np.array([3.0], dtype=np.float32),
            next_states=np.random.randn(1, 4).astype(np.float32),
            dones=np.array([False], dtype=np.bool_),
            env_idxs=np.array([0], dtype=np.int16),
        )

        result = deque.append(step_data)
        assert result is not None
        assert isinstance(result, _NStepData)
        assert result.states.shape == (1, 4)
        assert result.actions.shape == (1,)
        assert result.rewards.shape == (1,)
        assert result.n_next_states.shape == (1, 4)
        assert result.n_dones.shape == (1,)
        assert result.n.shape == (1,)
        assert result.n[0] == 3  # n-step

    def test_append_done_early(self) -> None:
        """Test appending data that ends with done=True."""
        deque = _NStepDeque(n_step=3, gamma=0.99)

        # Add first step
        step_data = _StepData(
            states=np.random.randn(1, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (1,)).astype(np.int64),
            rewards=np.array([1.0], dtype=np.float32),
            next_states=np.random.randn(1, 4).astype(np.float32),
            dones=np.array([False], dtype=np.bool_),
            env_idxs=np.array([0], dtype=np.int16),
        )

        result = deque.append(step_data)
        assert result is None

        # Add second step with done=True
        step_data = _StepData(
            states=np.random.randn(1, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (1,)).astype(np.int64),
            rewards=np.array([2.0], dtype=np.float32),
            next_states=np.random.randn(1, 4).astype(np.float32),
            dones=np.array([True], dtype=np.bool_),  # Done
            env_idxs=np.array([0], dtype=np.int16),
        )

        result = deque.append(step_data)
        assert result is not None
        assert isinstance(result, _NStepData)
        assert result.n[0] == 2  # 2-step return

    def test_append_multiple_envs(self) -> None:
        """Test appending data for multiple environments."""
        deque = _NStepDeque(n_step=2, gamma=0.99)

        # Add data for 2 environments
        step_data = _StepData(
            states=np.random.randn(2, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (2,)).astype(np.int64),
            rewards=np.array([1.0, 2.0], dtype=np.float32),
            next_states=np.random.randn(2, 4).astype(np.float32),
            dones=np.array([False, False], dtype=np.bool_),
            env_idxs=np.array([0, 1], dtype=np.int16),
        )

        result = deque.append(step_data)
        assert result is None

        # Add second step
        step_data = _StepData(
            states=np.random.randn(2, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (2,)).astype(np.int64),
            rewards=np.array([3.0, 4.0], dtype=np.float32),
            next_states=np.random.randn(2, 4).astype(np.float32),
            dones=np.array([False, False], dtype=np.bool_),
            env_idxs=np.array([0, 1], dtype=np.int16),
        )

        result = deque.append(step_data)
        assert result is not None
        assert isinstance(result, _NStepData)
        assert result.states.shape == (2, 4)  # 2 environments
        assert result.n[0] == 2  # 2-step return


class TestPERBuffer:
    """Test cases for the PERBuffer class."""

    def test_init(self) -> None:
        """Test PERBuffer initialization."""
        config = PERBufferConfig(
            capacity=100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )
        buffer = PERBuffer(config)

        assert buffer._config == config
        assert buffer._max_priority == 1.0
        assert buffer._beta == config.beta
        assert len(buffer) == 0

    def test_add_batch_empty(self) -> None:
        """Test adding empty batch."""
        config = PERBufferConfig(
            capacity=100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )
        buffer = PERBuffer(config)

        # Add empty batch
        buffer.add_batch(
            states=np.array([]),
            actions=np.array([]),
            rewards=np.array([]),
            next_states=np.array([]),
            dones=np.array([]),
            env_idxs=np.array([]),
        )

        assert len(buffer) == 0

    def test_add_batch_single_step(self) -> None:
        """Test adding single step batch."""
        config = PERBufferConfig(
            capacity=100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )
        buffer = PERBuffer(config)

        # Add single step (not enough for n-step)
        buffer.add_batch(
            states=np.random.randn(1, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (1,)).astype(np.int64),
            rewards=np.array([1.0], dtype=np.float32),
            next_states=np.random.randn(1, 4).astype(np.float32),
            dones=np.array([False], dtype=np.bool_),
            env_idxs=np.array([0], dtype=np.int16),
        )

        assert len(buffer) == 0  # Not enough steps for n-step return

    def test_add_batch_n_step(self) -> None:
        """Test adding enough steps for n-step return."""
        config = PERBufferConfig(
            capacity=100,
            n_step=2,
            gamma=0.99,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )
        buffer = PERBuffer(config)

        # Add first step
        buffer.add_batch(
            states=np.random.randn(1, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (1,)).astype(np.int64),
            rewards=np.array([1.0], dtype=np.float32),
            next_states=np.random.randn(1, 4).astype(np.float32),
            dones=np.array([False], dtype=np.bool_),
            env_idxs=np.array([0], dtype=np.int16),
        )

        # Add second step (should trigger n-step return)
        buffer.add_batch(
            states=np.random.randn(1, 4).astype(np.float32),
            actions=np.random.randint(0, 2, (1,)).astype(np.int64),
            rewards=np.array([2.0], dtype=np.float32),
            next_states=np.random.randn(1, 4).astype(np.float32),
            dones=np.array([False], dtype=np.bool_),
            env_idxs=np.array([0], dtype=np.int16),
        )

        assert len(buffer) == 1  # One n-step experience added

    def test_sample_empty(self) -> None:
        """Test sampling from empty buffer."""
        config = PERBufferConfig(
            capacity=100,
            n_step=2,
            gamma=0.99,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )
        buffer = PERBuffer(config)

        with pytest.raises(AssertionError, match="Cannot sample from an empty buffer"):
            buffer.sample(1)

    def test_sample_with_data(self) -> None:
        """Test sampling from buffer with data."""
        config = PERBufferConfig(
            capacity=100,
            n_step=2,
            gamma=0.99,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )
        buffer = PERBuffer(config)

        # Add data
        for i in range(3):
            buffer.add_batch(
                states=np.random.randn(1, 4).astype(np.float32),
                actions=np.random.randint(0, 2, (1,)).astype(np.int64),
                rewards=np.array([float(i)], dtype=np.float32),
                next_states=np.random.randn(1, 4).astype(np.float32),
                dones=np.array([False], dtype=np.bool_),
                env_idxs=np.array([0], dtype=np.int16),
            )

        # Sample
        replay, weights, indices = buffer.sample(2)

        assert isinstance(replay, NStepReplay)
        assert replay.states.shape == (2, 4)
        assert replay.actions.shape == (2,)
        assert replay.rewards.shape == (2,)
        assert replay.n_next_states.shape == (2, 4)
        assert replay.n_dones.shape == (2,)
        assert replay.n.shape == (2,)

        assert isinstance(weights, np.ndarray)
        assert weights.shape == (2,)
        assert weights.dtype == np.float32

        assert isinstance(indices, np.ndarray)
        assert indices.shape == (2,)
        assert indices.dtype == np.int64

    def test_update_priorities(self) -> None:
        """Test updating priorities."""
        config = PERBufferConfig(
            capacity=100,
            n_step=2,
            gamma=0.99,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )
        buffer = PERBuffer(config)

        # Add data
        for i in range(3):
            buffer.add_batch(
                states=np.random.randn(1, 4).astype(np.float32),
                actions=np.random.randint(0, 2, (1,)).astype(np.int64),
                rewards=np.array([float(i)], dtype=np.float32),
                next_states=np.random.randn(1, 4).astype(np.float32),
                dones=np.array([False], dtype=np.bool_),
                env_idxs=np.array([0], dtype=np.int16),
            )

        # Sample and update priorities
        replay, weights, indices = buffer.sample(2)
        new_priorities = np.array([0.5, 1.5], dtype=np.float32)

        buffer.update_priorities(indices, new_priorities)

        # Check that max priority was updated
        assert buffer._max_priority >= 1.5

    def test_update_priorities_mismatch_length(self) -> None:
        """Test updating priorities with mismatched lengths."""
        config = PERBufferConfig(
            capacity=100,
            n_step=2,
            gamma=0.99,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )
        buffer = PERBuffer(config)

        # Add data
        for i in range(3):
            buffer.add_batch(
                states=np.random.randn(1, 4).astype(np.float32),
                actions=np.random.randint(0, 2, (1,)).astype(np.int64),
                rewards=np.array([float(i)], dtype=np.float32),
                next_states=np.random.randn(1, 4).astype(np.float32),
                dones=np.array([False], dtype=np.bool_),
                env_idxs=np.array([0], dtype=np.int16),
            )

        # Sample
        replay, weights, indices = buffer.sample(2)

        # Try to update with mismatched lengths
        with pytest.raises(ValueError, match="idxs and priorities must have the same length"):
            buffer.update_priorities(indices, np.array([0.5]))

    def test_beta_schedule(self) -> None:
        """Test beta schedule increment."""
        config = PERBufferConfig(
            capacity=100,
            n_step=2,
            gamma=0.99,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.1,
        )
        buffer = PERBuffer(config)

        initial_beta = buffer._beta

        # Add data and sample multiple times
        for i in range(3):
            buffer.add_batch(
                states=np.random.randn(1, 4).astype(np.float32),
                actions=np.random.randint(0, 2, (1,)).astype(np.int64),
                rewards=np.array([float(i)], dtype=np.float32),
                next_states=np.random.randn(1, 4).astype(np.float32),
                dones=np.array([False], dtype=np.bool_),
                env_idxs=np.array([0], dtype=np.int16),
            )

        # Sample multiple times to trigger beta increment
        for _ in range(3):
            if len(buffer) > 0:
                buffer.sample(1)

        # Beta should have increased
        assert buffer._beta > initial_beta
        assert buffer._beta <= 1.0  # Should be capped at 1.0

    def test_n_step_return_calculation(self) -> None:
        """Test that n-step returns are calculated correctly."""
        config = PERBufferConfig(
            capacity=100,
            n_step=3,
            gamma=0.9,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
        )
        buffer = PERBuffer(config)

        # Add 3 steps with known rewards
        rewards = [1.0, 2.0, 3.0]
        for _, reward in enumerate(rewards):
            buffer.add_batch(
                states=np.random.randn(1, 4).astype(np.float32),
                actions=np.random.randint(0, 2, (1,)).astype(np.int64),
                rewards=np.array([reward], dtype=np.float32),
                next_states=np.random.randn(1, 4).astype(np.float32),
                dones=np.array([False], dtype=np.bool_),
                env_idxs=np.array([0], dtype=np.int16),
            )

        # Sample and check n-step return
        if len(buffer) > 0:
            replay, _, _ = buffer.sample(1)
            # The n-step return should be: 1 + 0.9*2 + 0.9^2*3 = 1 + 1.8 + 2.43 = 5.23
            expected_return = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0
            assert abs(replay.rewards[0].item() - expected_return) < 0.01
