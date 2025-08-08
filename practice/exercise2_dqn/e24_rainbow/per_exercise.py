from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from practice.utils_for_coding.replay_buffer_utils import Buffer


@dataclass(frozen=True, kw_only=True)
class PERBufferConfig:
    capacity: int
    """The capacity of the buffer."""
    n_step: int
    """The n-step return."""
    gamma: float
    """The discount factor."""
    alpha: float
    """The alpha parameter for prioritized experience replay."""
    beta: float
    """The beta parameter for prioritized experience replay."""
    beta_increment: float
    """The increment of beta per update."""


@dataclass(frozen=True, kw_only=True)
class NStepReplay:
    """The n-steps experience of the agent for a batch.

    Attributes:
        states:         State at the start  [batch, state_dim]
        actions:        Action taken        [batch, action_dim]
        rewards:        n-step reward       [batch]
        n_next_states:  Next n+1-state      [batch, state_dim]
        n_dones:        Done mask (0/1)     [batch]
        n:              The step n          [batch]
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    n_next_states: torch.Tensor
    n_dones: torch.Tensor
    n: torch.Tensor

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "NStepReplay":
        """Move all tensors to a device and/or dtype."""
        return NStepReplay(
            states=self.states.to(device, dtype),
            actions=self.actions.to(device, dtype),
            rewards=self.rewards.to(device, dtype),
            n_next_states=self.n_next_states.to(device, dtype),
            n_dones=self.n_dones.to(device, dtype),
            n=self.n.to(device, dtype),
        )


@dataclass(frozen=True, kw_only=True)
class _NStepData:
    """Data with the n-step returns."""

    states: NDArray[Any]
    actions: NDArray[Any]
    rewards: NDArray[np.float32]
    n_next_states: NDArray[Any]
    n_dones: NDArray[np.bool_]
    n: NDArray[np.int16]

    def to_replay(self) -> NStepReplay:
        return NStepReplay(
            states=torch.from_numpy(self.states),
            actions=torch.from_numpy(self.actions),
            rewards=torch.from_numpy(self.rewards),
            n_next_states=torch.from_numpy(self.n_next_states),
            n_dones=torch.from_numpy(self.n_dones),
            n=torch.from_numpy(self.n),
        )


@dataclass(frozen=True, kw_only=True)
class _StepData:
    """data for a single step."""

    states: NDArray[Any]
    actions: NDArray[Any]
    rewards: NDArray[np.float32]
    next_states: NDArray[Any]
    dones: NDArray[np.bool_]
    env_idxs: NDArray[np.int16]


class PERBuffer:
    """Prioritized Experience Replay Buffer with n-step return.

    Only support the single process case now, and follow the flow:
    >> .add_batch() -> .sample() -> .update_priorities() -> .add_batch() ...

    Reference:
    - https://arxiv.org/abs/1511.05952
    """

    def __init__(
        self,
        config: PERBufferConfig,
    ) -> None:
        self._config = config
        self._max_priority: float = 1.0
        self._sum_tree = _SumTree(config.capacity)
        self._replay_buffer = _ReplayBuffer(capacity=config.capacity)
        self._n_step_deque = _NStepDeque(n_step=config.n_step, gamma=config.gamma)
        # Internal beta schedule state (config is frozen)
        self._beta: float = config.beta

    def __len__(self) -> int:
        return len(self._replay_buffer)

    def add_batch(
        self,
        states: NDArray[Any],
        actions: NDArray[Any],
        rewards: NDArray[np.float32],
        next_states: NDArray[Any],
        dones: NDArray[np.bool_],
        env_idxs: NDArray[np.int16],
    ) -> None:
        if len(rewards) == 0:
            return

        n_step_data = self._n_step_deque.append(
            step_data=_StepData(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                env_idxs=env_idxs,
            )
        )
        if n_step_data is None:
            return

        # Write to underlying buffer and get written indices
        written_idxs = self._replay_buffer.add_batch(n_step_data=n_step_data)

        # Initialize priorities for newly written samples with current max priority
        # so that the new samples have a higher chance to be sampled and update priorities
        init_priorities = np.full(
            len(written_idxs), self._max_priority**self._config.alpha, dtype=np.float32
        )
        self._sum_tree.update(written_idxs, init_priorities)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[NStepReplay, NDArray[np.float32], NDArray[np.int64]]:
        """Sample a batch of experiences using the SumTree top-k.

        Returns:
            tuple[NStepReplay, NDArray[np.float32], NDArray[np.int32]]:
                - NStepReplay: The sampled experiences.
                - weights: Importance sampling weights for the sampled experiences. It will be used
                    to normalize the TD-errors.
                - indices: The sampled data indices in the underlying buffer.
        """
        assert len(self) > 0, "Cannot sample from an empty buffer"
        idxs, leaf_priorities = self._sum_tree.sample(batch_size)
        batch = self._replay_buffer.sample_by_idxs(idxs)

        # calculate weights for importance sampling
        # Handle the case where SumTree fell back to uniform sampling (total_priority = 0)
        if self._sum_tree.total_priority <= 0:
            # Uniform sampling case - all weights should be equal
            weights: NDArray[np.float32] = np.ones(len(idxs), dtype=np.float32)
        else:
            sampling_prob = leaf_priorities / (self._sum_tree.total_priority + 1e-8)
            # Update beta schedule
            self._beta = min(1.0, self._beta + self._config.beta_increment)
            weights = np.power(self._config.capacity * sampling_prob, -self._beta, dtype=np.float32)
            # Normalize weights to avoid NaN
            max_weight = np.max(weights)
            if max_weight > 0:
                weights /= max_weight + 1e-6
            else:
                weights = np.ones_like(weights)

        return batch.to_replay(), weights, idxs

    def update_priorities(
        self,
        idxs: NDArray[np.integer[Any]] | Sequence[int],
        priorities: NDArray[np.float32] | Sequence[float],
    ) -> None:
        """Update the priorities of the sampled experiences.

        The input indices must be the data indices in the underlying `ReplayBufferPro`.

        Args:
            idxs: Data indices of the sampled experiences.
            priorities: New priority values (TD-errors) for the sampled experiences.
        """
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        prios_arr = np.asarray(priorities, dtype=np.float32)
        if idxs_arr.shape[0] != prios_arr.shape[0]:
            raise ValueError("idxs and priorities must have the same length")

        # Track max priority for future insertions
        if prios_arr.size > 0:
            self._max_priority = max(self._max_priority, float(np.max(prios_arr)))
        # The SumTree stores alpha-powered priorities
        self._sum_tree.update(idxs_arr, np.power(prios_arr, self._config.alpha, dtype=np.float32))


class _SumTree:
    """SumTree for prioritized experience replay.

    SumTree is a binary tree where the parent node is the sum of the two child nodes. It makes
    sampling more efficient.
    - The leaf nodes are the priorities of the experiences.
    - The root node is the total priority of all experiences.

    Reference:
    - https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        # [0, Capacity) is the tree, [Capacity, 2*Capacity-1) is the data
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        # Track which leaves have been initialized (non-zero at least once)
        self._used = np.zeros(capacity, dtype=np.bool_)

        # pre-calculate variables
        self._tree_height = int(np.ceil(np.log2(self._capacity))) + 1
        self._capacity_minus_1 = self._capacity - 1

    def update(self, idxs: NDArray[np.integer[Any]], priorities: NDArray[np.float32]) -> None:
        """Batch update priorities."""
        # 1. update used flag
        newly_used_mask = ~self._used[idxs]
        if np.any(newly_used_mask):
            self._used[idxs[newly_used_mask]] = True

        # 2. calculate leaf indices and differences
        leaf_indices = idxs + self._capacity_minus_1
        old_priorities = self._tree[leaf_indices]
        diffs = priorities - old_priorities

        # 3. update leaf nodes
        self._tree[leaf_indices] = priorities

        # 4. update parent nodes level by level
        current_level = leaf_indices.copy()
        for _ in range(self._tree_height):
            # calculate parent nodes
            parents = (current_level - 1) // 2
            parents[parents < 0] = 0  # handle root node

            # stop if all parents are root
            if np.all(parents == 0):
                # update root node
                self._tree[0] += np.sum(diffs)
                break

            # group by parent and aggregate differences
            unique_parents, inverse_indices = np.unique(parents, return_inverse=True)
            parent_diffs = np.bincount(
                inverse_indices, weights=diffs, minlength=len(unique_parents)
            )

            # update parent nodes
            self._tree[unique_parents] += parent_diffs

            # prepare for next level
            current_level = unique_parents
            diffs = parent_diffs

    def sample(self, k: int) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Return the k data indices with stratified sampling.

        Returns:
            (idxs, priorities):
                - idxs: Data indices of the sampled data.
                - priorities: Their leaf priorities.
        """
        size = len(self)
        assert 0 < k <= size, f"the buffer size: {size}, but required: {k}"
        total_prio = self.total_priority
        if total_prio <= 0:
            # Fallback to uniform sampling if no priority
            valid_indices = np.where(self._used)[0]
            idxs = np.random.choice(valid_indices, size=k, replace=True)
            return idxs, self._tree[idxs + self._capacity_minus_1]

        segment = total_prio / k  # Stratified sampling interval
        # Generate stratified random values
        v_values = np.random.uniform(
            np.arange(k) * segment,
            (np.arange(k) + 1) * segment,
        ).astype(np.float32)

        # Use vectorized get_leaves method
        return self._get_leaves(v_values)

    def _get_leaves(
        self,
        v_values: NDArray[np.float32],
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Get multiple leaf nodes and data indices in a vectorized manner.

        Args:
            v_values: Array of values to get leaf nodes for.

        Returns:
            (data_indices, priorities):
                - data_indices: The indices of the data in the buffer.
                - priorities: The priorities of the leaf nodes.
        """
        batch_size = len(v_values)
        if batch_size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Initialize arrays for batch processing
        leaf_idxs = np.zeros(batch_size, dtype=np.int64)

        # Vectorized tree traversal
        # We need to traverse the tree level by level for all values simultaneously

        for level in range(self._tree_height):
            # Check if we've reached the leaf level
            if level == self._tree_height - 1:
                break

            # Calculate left and right child indices for all current positions
            left_children = 2 * leaf_idxs + 1
            right_children = left_children + 1

            # Check which path to take for each value
            # We need to be careful about bounds checking
            valid_mask = left_children < len(self._tree)

            if not np.any(valid_mask):
                break

            # Get left child values where valid
            left_values = np.where(valid_mask, self._tree[left_children], 0.0)

            # Determine which path to take
            go_left = (v_values <= left_values) & valid_mask
            go_right = ~go_left & valid_mask

            # Update indices and values
            leaf_idxs[go_left] = left_children[go_left]
            leaf_idxs[go_right] = right_children[go_right]
            v_values[go_right] -= left_values[go_right]

        # Calculate data indices and get priorities
        data_indices = leaf_idxs - self._capacity_minus_1
        priorities = self._tree[leaf_idxs]

        return data_indices, priorities

    @property
    def total_priority(self) -> float:
        return float(self._tree[0])

    def __len__(self) -> int:
        """The number of data in the buffer."""
        return int(np.sum(self._used))


class _ReplayBuffer:
    """A replay buffer for any type of data."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buffer = Buffer(capacity=capacity)

    def add_batch(self, n_step_data: _NStepData) -> NDArray[np.int64]:
        # get indices for batch writing
        return self._buffer.add_batch(
            states=n_step_data.states,
            actions=n_step_data.actions,
            rewards=n_step_data.rewards,
            n_next_states=n_step_data.n_next_states,
            n_dones=n_step_data.n_dones,
            n=n_step_data.n,
        )

    def sample(self, batch_size: int) -> _NStepData:
        """Sample a batch of data from the buffer."""
        idxs = np.random.randint(0, len(self), batch_size)
        return self.sample_by_idxs(idxs)

    def sample_by_idxs(self, idxs: Sequence[int] | NDArray[np.integer[Any]]) -> _NStepData:
        """Sample a batch of data from the buffer by indices."""
        assert len(idxs) <= len(self), "the number of indices should <= buffer size"
        idxs_arr: NDArray[np.int64] = np.asarray(idxs, dtype=np.int64)
        batch = self._buffer.sample_by_idxs(idxs_arr)
        return _NStepData(
            states=batch["states"].numpy(),
            actions=batch["actions"].numpy(),
            rewards=batch["rewards"].numpy(),
            n_next_states=batch["n_next_states"].numpy(),
            n_dones=batch["n_dones"].numpy(),
            n=batch["n"].numpy(),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class _NStepDeque:
    """A deque for n-step return.

    After enqueuing a new item, will compute n-step returns and return the data from deques if:
    1. a deque is full
    2. the last/latest data of a deque is done
    """

    def __init__(self, n_step: int, gamma: float) -> None:
        self._n_step = n_step
        self._gamma = gamma
        self._init_flag = False

    def _init_if_needed(self, num_envs: int) -> None:
        if self._init_flag:
            return

        self._deques: tuple[deque[_StepData], ...] = tuple(
            deque(maxlen=self._n_step) for _ in range(num_envs)
        )
        self._init_flag = True

    def append(self, step_data: _StepData) -> _NStepData | None:
        self._init_if_needed(num_envs=len(step_data.rewards))
        # add data to deques
        for idx, env_idx in enumerate(step_data.env_idxs):
            env_data = _StepData(
                # use idx: idx + 1 to keep the batch dimension
                states=step_data.states[idx : idx + 1],
                actions=step_data.actions[idx : idx + 1],
                rewards=step_data.rewards[idx : idx + 1],
                next_states=step_data.next_states[idx : idx + 1],
                dones=step_data.dones[idx : idx + 1],
                env_idxs=step_data.env_idxs[idx : idx + 1],
            )
            self._deques[env_idx].append(env_data)
        return self._process_n_step_data()

    def _process_n_step_data(self) -> _NStepData | None:
        """Compute the n-step return for all queues."""
        data_list: list[_NStepData] = []
        for env_idx in range(len(self._deques)):
            data = self._process_one_queue(env_idx)
            if data is not None:
                data_list.append(data)
        return _concat_n_step_data(data_list) if data_list else None

    def _process_one_queue(self, env_idx: int) -> _NStepData | None:
        """Process one queue and return the data from deques if:
        1. a deque is full
        2. the last/latest data of a deque is done
        """
        env_deque = self._deques[env_idx]
        # If deque is empty, return None
        if len(env_deque) == 0:
            return None

        is_full = len(env_deque) >= self._n_step
        latest_data = env_deque[-1]
        assert latest_data.dones.size == 1, "the number of dones should be 1"
        is_latest_done = bool(np.any(latest_data.dones))
        if not (is_full or is_latest_done):
            return None

        # handle the case when the latest data is done
        if is_latest_done:
            data_list: list[_NStepData] = []
            while len(env_deque) >= 1:
                # _get_one_n_step_data will pop the oldest data from the deque
                data_list.append(self._get_one_n_step_data(env_idx, is_latest_done))
            return _concat_n_step_data(data_list)

        # handle the case when the deque is full
        assert is_full, "the deque is not full"
        # compute n-step return
        return self._get_one_n_step_data(env_idx, is_latest_done=False)

    def _get_one_n_step_data(self, env_idx: int, is_latest_done: bool) -> _NStepData:
        """Get one n-step data from the deque.

        !!Careful, only call this function when the deque is full and the latest data is done!!

        It will pop the oldest data from the deque.
        """
        env_deque = self._deques[env_idx]
        # compute n-step return
        n_step_return = np.zeros_like(env_deque[0].rewards)
        n_step = len(env_deque)
        for data in reversed(env_deque):
            n_step_return = self._gamma * n_step_return + data.rewards
            if not is_latest_done:
                assert not np.any(data.dones), "it should process all data when is_last_done"

        oldest_data = env_deque.popleft()

        # If this is the last item in the deque, use the current data as next_states
        if len(env_deque) == 0:
            # This is the last item, so use the current data as next_states
            next_states = oldest_data.next_states
            next_dones = oldest_data.dones
        else:
            # Use the remaining data in the deque
            next_states = env_deque[-1].next_states
            next_dones = env_deque[-1].dones

        return _NStepData(
            states=oldest_data.states,
            actions=oldest_data.actions,
            rewards=n_step_return,
            n_next_states=next_states,
            n_dones=next_dones,
            n=np.array([n_step], dtype=np.int16),
        )


def _concat_n_step_data(data_list: list[_NStepData]) -> _NStepData:
    """Concatenate n-step data.

    Args:
        data_list: A list of n-step data.

    Returns:
        The concatenated n-step data.
    """
    return _NStepData(
        states=np.concatenate([data.states for data in data_list], axis=0),
        actions=np.concatenate([data.actions for data in data_list], axis=0),
        rewards=np.concatenate([data.rewards for data in data_list], axis=0),
        n_next_states=np.concatenate([data.n_next_states for data in data_list], axis=0),
        n_dones=np.concatenate([data.n_dones for data in data_list], axis=0),
        n=np.concatenate([data.n for data in data_list], axis=0),
    )
