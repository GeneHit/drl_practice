from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from practice.exercise2_dqn.e24_rainbow.sum_tree import SumTree
from practice.utils_for_coding.replay_buffer_utils import Buffer


@dataclass(frozen=True, kw_only=True)
class PERBufferConfig:
    capacity: int
    """The capacity of the buffer."""
    n_step: int
    """The n-step return."""
    gamma: float
    """The discount factor."""
    use_uniform_sampling: bool = False
    """Whether to use uniform sampling instead of prioritized sampling."""
    alpha: float
    """The alpha parameter for prioritized experience replay."""
    beta: float
    """The beta parameter for prioritized experience replay."""
    beta_increment: float
    """The increment of beta per update."""

    def __post_init__(self) -> None:
        assert self.capacity > 1, f"{self.capacity=} must be greater than 1"
        assert self.n_step > 0, f"{self.n_step=} must be greater than 0"
        assert 0.0 <= self.beta <= 1.0, f"{self.beta=} must be in [0.0, 1.0]"
        assert 0.0 <= self.beta_increment <= 1.0, f"{self.beta_increment=} must be in [0.0, 1.0]"
        assert 0.0 <= self.alpha <= 1.0, f"{self.alpha=} must be in [0.0, 1.0]"


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

    def __init__(self, config: PERBufferConfig) -> None:
        self._config = config
        self._sum_tree = SumTree(config.capacity)
        self._replay_buffer = _ReplayBuffer(capacity=config.capacity)
        self._n_step_deque = _NStepDeque(n_step=config.n_step, gamma=config.gamma)

        # Internal beta schedule state (config is frozen)
        self._beta: float = config.beta
        # the value for quantile clipping
        self._max_p_alpha = 1.0
        self._p95_alpha = 0.5
        self._epsilon_alpha = 1e-3

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
        # n_step == 1 is a special case, we can directly add the data to the buffer
        if self._config.n_step == 1:
            self._replay_buffer.add_batch(
                n_step_data=_NStepData(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    n_next_states=next_states,
                    n_dones=dones,
                    n=np.ones_like(env_idxs, dtype=np.int16),
                )
            )
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

        # Write the n-step return to underlying buffer and get written indices
        written_idxs = self._replay_buffer.add_batch(n_step_data=n_step_data)
        if self._config.use_uniform_sampling:
            return

        # Initialize priorities with quantile clipping (or fixed value), making the new samples
        # have a higher chance to be sampled and update priorities, but not too big.
        p0_alpha = np.clip(2 * self._p95_alpha, self._epsilon_alpha, self._max_p_alpha)
        init_priorities = np.full(len(written_idxs), p0_alpha, dtype=np.float32)
        self._sum_tree.update(written_idxs, init_priorities)

    def sample(self, batch_size: int) -> tuple[NStepReplay, NDArray[np.float32], NDArray[np.int64]]:
        """Sample a batch of experiences using the SumTree top-k.

        Returns:
            tuple[NStepReplay, NDArray[np.float32], NDArray[np.int32]]:
                - NStepReplay: The sampled experiences.
                - weights: Importance sampling weights for the sampled experiences. It will be used
                    to normalize the TD-errors.
                - indices: The sampled data indices in the underlying buffer.
        """
        assert len(self) >= batch_size > 0, f"buffer size {len(self)}, but required {batch_size}"

        # use uniform sampling if configured or necessary
        tp = self._sum_tree.total_priority
        uniform_sampling = self._config.use_uniform_sampling or tp < 1e-4 or not np.isfinite(tp)
        if uniform_sampling:
            idxs = np.random.randint(0, len(self), batch_size)
            # Uniform sampling case - all weights should be equal
            return (
                self._replay_buffer.sample_by_idxs(idxs).to_replay(),
                np.ones(batch_size, dtype=np.float32),
                idxs,
            )

        # sample from sum tree with proportional priority
        idxs, leaf_priorities = self._sum_tree.sample(batch_size)
        batch = self._replay_buffer.sample_by_idxs(idxs).to_replay()

        # calculate importance sampling weights
        sampling_prob = leaf_priorities.astype(np.float64) / (tp + 1e-8)
        self._beta = min(1.0, self._beta + self._config.beta_increment)
        n = len(self)
        weights = np.power(n * sampling_prob, -self._beta)
        weights = weights / (weights.max() + 1e-8)  # avoid zero division

        return batch, weights.astype(np.float32), idxs

    def update_priorities(
        self,
        idxs: NDArray[np.integer[Any]] | Sequence[int],
        priorities: NDArray[np.float32] | Sequence[float],
        get_metrics: bool = False,
    ) -> dict[str, float]:
        """Update the priorities of the sampled experiences.

        The input indices must be the data indices in the underlying `ReplayBufferPro`.

        Args:
            idxs: Data indices of the sampled experiences.
            priorities: New priority values (TD-errors) for the sampled experiences.
            get_metrics: Whether to get the metrics of the buffer.

        Returns:
            dict[str, float]: The metrics of the buffer.
                If `get_metrics` is False, return {}.
        """
        if self._config.use_uniform_sampling:
            return {}

        idxs_arr = np.asarray(idxs, dtype=np.int64)
        prios_arr = np.asarray(priorities, dtype=np.float32)
        if idxs_arr.shape[0] != prios_arr.shape[0]:
            raise ValueError("idxs and priorities must have the same length")

        # update priorities in sum tree
        prio_raw = np.abs(prios_arr) + self._epsilon_alpha  # epsilon to avoid zero priority
        p_alpha = np.power(prio_raw, self._config.alpha, dtype=np.float32)
        self._sum_tree.update(idxs_arr, p_alpha)

        # track max and percentile for future insertions
        decay = 0.99
        self._max_p_alpha = max(self._max_p_alpha, np.max(p_alpha))
        self._p95_alpha = decay * self._p95_alpha + (1 - decay) * np.quantile(p_alpha, 0.95)

        if get_metrics:
            return {
                "per/beta": self._beta,
                # batch diversity: higher is better
                "per/unique_ratio": len(np.unique(idxs_arr)) / len(idxs_arr),
                **self._sum_tree.tree_metrics(),
            }
        return {}


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
            states=batch["states"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            n_next_states=batch["n_next_states"],
            n_dones=batch["n_dones"],
            n=batch["n"],
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
