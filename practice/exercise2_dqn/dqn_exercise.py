import abc
import copy
from dataclasses import dataclass
from typing import Literal, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Discrete
from numpy.typing import NDArray
from torch import Tensor

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ObsType
from practice.utils_for_coding.network_utils import init_weights
from practice.utils_for_coding.numpy_tensor_utils import argmax_action
from practice.utils_for_coding.replay_buffer_utils import ReplayBuffer
from practice.utils_for_coding.scheduler_utils import ScheduleBase
from practice.utils_for_coding.writer_utils import CustomWriter


class QNet1D(nn.Module):
    """Q network with 1D discrete observation space."""

    def __init__(
        self,
        state_n: int,
        action_n: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        pre_size = state_n
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(pre_size, hidden_size))
            layers.append(nn.ReLU())
            pre_size = hidden_size
        layers.append(nn.Linear(pre_size, action_n))
        self.network = nn.Sequential(*layers)
        self.network.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use cast to make mypy happy
        return cast(Tensor, self.network(x))

    def action(self, x: Tensor) -> ActType:
        """Get the action for evaluation/gameplay with 1 environment.

        Returns:
            action: The single action.
        """
        # greedy strategy
        return argmax_action(self.forward(x), dtype=ActType)


def get_dqn_actions(
    network: nn.Module,
    states: NDArray[ObsType],
    epsilon: float,
    env_state_shape: tuple[int, ...],
    action_n: int,
    step: int,
    writer: CustomWriter,
    log_interval: int,
    device: torch.device,
) -> NDArray[ActType]:
    """Get action(s) for state(s) with epsilon-greedy strategy.

    Steps:
    1. ensure states is a batch of states
    2. get random actions (exploration) for partial environments
    3. get greedy actions (exploitation) for the rest of the environments
    4. log stats: epsilon, action mean & std.
    5. return actions

    Args:
        network: The network to use for action selection.
        state: Single state or batch of states
        epsilon: The epsilon value for exploration.
        env_state_shape: The shape of the state space.
        action_n: The number of actions.
        step: The current step.
        writer: The writer to use for logging.
        log_interval: The interval for logging.
        device: The device to use for computation.

    Returns:
        actions: NDArray[ActType]
            Single action or batch of actions depending on input shape.
    """
    # Check if input is a single state or batch of states
    is_single = len(states.shape) == len(env_state_shape)
    state_batch = states if not is_single else states.reshape(1, *states.shape)

    # Training phase: epsilon-greedy
    batch_size = state_batch.shape[0]
    actions = np.zeros(batch_size, dtype=ActType)

    # Random mask for exploration
    random_mask = np.random.random(batch_size) < epsilon
    # Random actions for exploration
    num_random = int(np.sum(random_mask))
    actions[random_mask] = np.random.randint(0, action_n, size=num_random, dtype=ActType)

    # Greedy actions for exploitation
    if not np.all(random_mask):
        exploit_states = state_batch[~random_mask]
        state_tensor = torch.from_numpy(exploit_states).to(device)
        with torch.no_grad():
            q_values = network(state_tensor).cpu()
            greedy_actions = q_values.argmax(dim=1).numpy().astype(ActType)
            actions[~random_mask] = greedy_actions

    # Log stats
    writer.log_stats(
        data={
            "action/epsilon": epsilon,
            "action/mean": actions.mean(),
            "action/std": actions.std(),
        },
        step=step,
        log_interval=log_interval,
    )
    return cast(NDArray[ActType], actions)


class DQNPod(abc.ABC):
    """Abstract base class for DQN training pod."""

    @abc.abstractmethod
    def __init__(self, config: BaseConfig, ctx: ContextBase, writer: CustomWriter) -> None: ...

    @abc.abstractmethod
    def sync_target_net(self) -> None:
        """Synchronize target network with current Q-network."""

    @abc.abstractmethod
    def action(self, states: NDArray[ObsType]) -> NDArray[ActType]:
        """Get action(s) for state(s) when training."""

    @abc.abstractmethod
    def update(self) -> None:
        """Update Q-network using the inside replay buffer."""

    @abc.abstractmethod
    def buffer_add(
        self,
        states: NDArray[ObsType],
        actions: NDArray[ActType],
        rewards: NDArray[np.float32],
        next_states: NDArray[ObsType],
        dones: NDArray[np.bool_],
        env_idxs: NDArray[np.int16],
    ) -> None:
        """Add batch of experiences to the replay buffer."""


@dataclass(kw_only=True, frozen=True)
class DQNConfig(BaseConfig):
    """Configuration for DQN algorithm."""

    timesteps: int
    """The loop step number to train the policy.

    The total_data = timesteps * vector_env_num.
    """
    epsilon_schedule: ScheduleBase
    """The epsilon schedule for the DQN algorithm."""
    replay_buffer_capacity: int
    """The capacity of the replay buffer."""
    batch_size: int
    """The batch size for the DQN algorithm."""
    train_interval: int
    """The interval for training the DQN algorithm."""
    target_update_interval: int
    """The interval for updating the target network."""
    update_start_step: int
    """The step number to start updating the target network."""
    dqn_algorithm: Literal["basic", "double", "rainbow"]
    """The DQN algorithm to use."""


class BasicDQNPod(DQNPod):
    """Internal pod for DQN training logic."""

    def __init__(self, config: DQNConfig, ctx: ContextBase, writer: CustomWriter) -> None:
        # Notice: can use many variables from super class
        self._config = config
        self._ctx = ctx
        self._writer = writer

        # Create target network
        self._target_net = copy.deepcopy(ctx.network)
        self._target_net.eval()
        self._ctx.network.train()

        # Create replay buffer
        self._replay = ReplayBuffer(
            capacity=self._config.replay_buffer_capacity,
            state_shape=self._ctx.env_state_shape,
            state_dtype=np.float32,
            action_dtype=np.int64,
        )

        # Get action space info
        assert isinstance(self._ctx.eval_env.action_space, Discrete)
        self._action_n = int(self._ctx.eval_env.action_space.n)
        self._step = 0

    def sync_target_net(self) -> None:
        """Synchronize target network with current Q-network."""
        self._target_net.load_state_dict(self._ctx.network.state_dict())

    def buffer_add(
        self,
        states: NDArray[ObsType],
        actions: NDArray[ActType],
        rewards: NDArray[np.float32],
        next_states: NDArray[ObsType],
        dones: NDArray[np.bool_],
        env_idxs: NDArray[np.int16],
    ) -> None:
        """Add batch of experiences to the replay buffer."""
        self._replay.add_batch(states, actions, rewards, next_states, dones)

    def action(self, states: NDArray[ObsType]) -> NDArray[ActType]:
        """Get action(s) for state(s).

        Args:
            state: Single state or batch of states

        Returns:
            actions: NDArray[ActType]
                Single action or batch of actions depending on input shape.
        """
        actions = get_dqn_actions(
            network=self._ctx.network,
            states=states,
            epsilon=self._config.epsilon_schedule(self._step),
            env_state_shape=self._ctx.env_state_shape,
            action_n=self._action_n,
            step=self._step,
            writer=self._writer,
            log_interval=self._config.log_interval,
            device=self._config.device,
        )
        self._step += 1
        return actions

    def update(self) -> None:
        """Update Q-network using experiences.

        Args:
            experiences: Batch of experiences from replay buffer

        Returns:
            loss: The TD loss value for logging
        """
        if len(self._replay) < self._config.batch_size:
            return

        # sample batch and move to device
        exps = self._replay.sample(self._config.batch_size).to(self._config.device)

        # Compute TD target using target network
        with torch.no_grad():
            target_max: Tensor = self._target_net(exps.next_states).max(dim=1)[0]
            td_target = exps.rewards.flatten() + self._config.gamma * target_max * (
                1 - exps.dones.flatten().float()
            )

        # Get current Q-values for the actions taken
        current_q = self._ctx.network(exps.states).gather(1, exps.actions.view(-1, 1)).squeeze()

        # Compute loss and update
        self._ctx.optimizer.zero_grad()
        loss = F.mse_loss(current_q, td_target)
        loss.backward()
        self._ctx.optimizer.step()

        self._writer.log_stats(
            data={"loss/td_loss": loss},
            step=self._step,
            log_interval=self._config.log_interval,
            blocked=False,
        )
