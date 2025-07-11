"""Train the DQN agent.

Reference:
Algorithm: https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm
Code:https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
"""

import copy
from typing import TypeAlias, Union, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from gymnasium.spaces import Discrete
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from hands_on.base import ActType, AgentBase, ScheduleBase
from hands_on.exercise2_dqn.config import DQNTrainConfig
from hands_on.utils.env_utils import extract_episode_data_from_infos
from hands_on.utils_for_coding.numpy_tensor_utils import (
    get_tensor_expanding_axis,
)
from hands_on.utils_for_coding.replay_buffer_utils import (
    Experience,
    ReplayBuffer,
)
from hands_on.utils_for_coding.scheduler_utils import LinearSchedule

ObsType: TypeAlias = Union[np.uint8, np.float32]
ArrayType: TypeAlias = Union[np.bool_, np.float32]
EnvType: TypeAlias = gym.Env[NDArray[ObsType], ActType]
EnvsType: TypeAlias = gym.vector.VectorEnv[NDArray[ObsType], ActType, NDArray[ArrayType]]


class QNet2D(nn.Module):
    """Q network with 2D convolution.

    Same as DeepMind's DQN paper for Atari:
    https://www.nature.com/articles/nature14236
    """

    def __init__(self, in_shape: tuple[int, int, int], action_n: int) -> None:
        super().__init__()
        c, h, w = in_shape
        # convolution layer sequence
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # calculate the output size of the convolution layer
        with torch.no_grad():
            # create a mock input (batch=1, c, h, w)
            test_input = torch.zeros(1, c, h, w)
            conv_output = self.conv(test_input)
            conv_output_size = conv_output.size(1)

        # full connected layer, original 512
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(self.conv(x / 255.0))
        assert isinstance(y, torch.Tensor)  # make mypy happy
        return y


class QNet1D(nn.Module):
    """Q network with 1D discrete observation space."""

    def __init__(self, state_n: int, action_n: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_n, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x)
        assert isinstance(y, torch.Tensor)  # make mypy happy
        return y


class DQNAgent(AgentBase):
    """DQN agent for evaluation/gameplay.

    This agent is focused on action selection using a trained Q-network.
    It does not handle training-specific operations.
    """

    def __init__(self, q_network: nn.Module) -> None:
        self._q_network = q_network
        self._device = next(q_network.parameters()).device

    def action(self, state: NDArray[ObsType]) -> ActType:
        """Get action for single state using greedy policy."""
        # Always use greedy policy for trained agent evaluation
        self._q_network.eval()
        state_tensor = get_tensor_expanding_axis(state).to(self._device)
        with torch.no_grad():
            q_values = self._q_network(state_tensor).cpu()
        return ActType(q_values.argmax().item())

    def only_save_model(self, pathname: str) -> None:
        """Save the DQN model."""
        assert pathname.endswith(".pth")
        torch.save(self._q_network, pathname)

    @classmethod
    def load_from_checkpoint(cls, pathname: str, device: torch.device | None) -> "DQNAgent":
        """Load the DQN model."""
        assert pathname.endswith(".pth")
        q_network = torch.load(pathname, map_location=device, weights_only=False)
        q_network = q_network.to(device)
        return cls(q_network=q_network)


class DQNTrainer:
    """Handles DQN training operations."""

    def __init__(
        self,
        q_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gamma: float,
        epsilon: ScheduleBase,
        state_shape: tuple[int, ...],
        action_n: int,
    ) -> None:
        self._q_network = q_network
        self._target_net = copy.deepcopy(q_network)
        self._target_net.eval()
        self._optimizer = optimizer
        self._device = device
        self._gamma = gamma
        self._epsilon = epsilon
        self._state_shape = state_shape
        self._action_n = action_n

    def sync_target_net(self) -> None:
        """Synchronize target network with current Q-network."""
        self._target_net.load_state_dict(self._q_network.state_dict())

    def action(self, state: NDArray[ObsType], step: int, eval: bool = False) -> NDArray[ActType]:
        """Get action(s) for state(s).

        Args:
            state: Single state or batch of states
            step: Current step in the training process
            eval: If True, use greedy policy (epsilon=0)

        Returns:
            actions: NDArray[ActType]
                Single action or batch of actions depending on input shape.
                If the input is a single state, output a (1, ) array, making
                the output type consistent.
        """
        # Check if input is a single state or batch of states
        is_single = len(state.shape) == len(self._state_shape)
        state_batch = state if not is_single else state.reshape(1, *state.shape)

        if eval:
            # Evaluation phase: always greedy
            state_tensor = torch.from_numpy(state_batch).to(self._device)
            with torch.no_grad():
                q_values = self._q_network(state_tensor).cpu()
                actions = q_values.argmax(dim=1).numpy().astype(ActType)
            return cast(NDArray[ActType], actions)

        # Training phase: epsilon-greedy
        batch_size = state_batch.shape[0]
        actions = np.zeros(batch_size, dtype=ActType)

        # Random mask for exploration
        epsilon = self._epsilon(step)
        random_mask = np.random.random(batch_size) < epsilon
        # Random actions for exploration
        num_random = int(np.sum(random_mask))
        actions[random_mask] = np.random.randint(0, self._action_n, size=num_random, dtype=ActType)

        # Greedy actions for exploitation
        if not np.all(random_mask):
            exploit_states = state_batch[~random_mask]
            state_tensor = torch.from_numpy(exploit_states).to(self._device)
            self._q_network.train()
            with torch.no_grad():
                q_values = self._q_network(state_tensor).cpu()
                greedy_actions = q_values.argmax(dim=1).numpy().astype(ActType)
                actions[~random_mask] = greedy_actions

        return cast(NDArray[ActType], actions)

    def update(self, experiences: Experience) -> float:
        """Update Q-network using experiences.

        Args:
            experiences: Batch of experiences from replay buffer

        Returns:
            loss: The TD loss value for logging
        """
        # Move all inputs to device
        states = experiences.states.to(self._device)
        actions = experiences.actions.view(-1, 1).to(self._device)
        rewards = experiences.rewards.to(self._device)
        next_states = experiences.next_states.to(self._device)
        dones = experiences.dones.to(self._device)

        # Compute TD target using target network
        with torch.no_grad():
            target_max: Tensor = self._target_net(next_states).max(dim=1)[0]
            td_target = rewards.flatten() + self._gamma * target_max * (1 - dones.flatten().float())

        # Get current Q-values for the actions taken
        self._q_network.train()
        current_q = self._q_network(states).gather(1, actions).squeeze()

        # Compute loss and update
        self._optimizer.zero_grad()
        loss = F.mse_loss(current_q, td_target)
        loss.backward()
        self._optimizer.step()

        return float(loss.item())


def dqn_train_loop(
    envs: EnvsType,
    q_network: nn.Module,
    device: torch.device,
    config: DQNTrainConfig,
    log_dir: str,
) -> None:
    """Train the DQN agent with multiple environments.

    Args:
        envs: Vector environment for training
        q_network: The Q-network to train
        device: Device to run computations on
        config: Training configuration
        log_dir: Directory for tensorboard logs.
    """
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)

    # Get environment info
    obs_shape = envs.single_observation_space.shape
    act_space = envs.single_action_space
    assert obs_shape is not None
    assert isinstance(act_space, Discrete)
    action_n: int = int(act_space.n)

    epsilon_schedule = LinearSchedule(
        start_e=config.start_epsilon,
        end_e=config.end_epsilon,
        duration=int(config.exploration_fraction * config.timesteps),
    )

    # Create trainer inside the function
    trainer = DQNTrainer(
        q_network=q_network,
        optimizer=torch.optim.Adam(q_network.parameters(), lr=config.learning_rate),
        device=device,
        gamma=config.gamma,
        epsilon=epsilon_schedule,
        state_shape=obs_shape,
        action_n=action_n,
    )

    # Get state shape from environment
    replay_buffer = ReplayBuffer(
        capacity=config.replay_buffer_capacity,
        state_shape=obs_shape,
        state_dtype=np.float32,
        action_dtype=np.int64,
    )

    # Initialize environments
    states, _ = envs.reset()
    assert isinstance(states, np.ndarray), "States must be numpy array"
    # Track previous step terminal status to avoid invalid transitions
    prev_dones: NDArray[np.bool_] = np.zeros(envs.num_envs, dtype=np.bool_)
    episode_steps = 0

    for step in tqdm.tqdm(range(config.timesteps)):
        # Get actions for all environments using trainer's epsilon-greedy action method
        actions = trainer.action(states, step)

        # Step all environments
        next_states, rewards, terms, truncs, infos = envs.step(actions.tolist())

        # Handle terminal observations and create proper training transitions
        dones = np.logical_or(terms, truncs, dtype=np.bool_)

        # Only store transitions for states that were not terminal in the previous step
        # When an episode ends, the next state is a reset state from autoreset wrapper,
        # which creates invalid transitions that shouldn't be learned from
        pre_non_terminal_mask = ~prev_dones
        if np.any(pre_non_terminal_mask):
            # Only store transitions where the previous step didn't end an episode
            replay_buffer.add_batch(
                states=states[pre_non_terminal_mask],
                actions=actions[pre_non_terminal_mask],
                rewards=rewards[pre_non_terminal_mask],
                next_states=next_states[pre_non_terminal_mask].copy(),
                dones=dones[pre_non_terminal_mask],
            )

        # Update state and previous done status for next iteration
        states = next_states
        prev_dones = dones

        # Training updates
        if step >= config.update_start_step:
            if step % config.train_interval == 0:
                experiences = replay_buffer.sample(config.batch_size)
                loss = trainer.update(experiences)

                # Log training metrics
                writer.add_scalar("training/loss", loss, step)
                writer.add_scalar("training/epsilon", epsilon_schedule(step), step)

            if step % config.target_update_interval == 0:
                trainer.sync_target_net()

        # get the episode rewards from RecordEpisodeStatistics wrapper
        # Use the new utility function to extract episode data
        ep_rewards, ep_lengths = extract_episode_data_from_infos(infos)

        # Log episode metrics when episodes complete
        for idx, reward in enumerate(ep_rewards):
            writer.add_scalar("episode/reward", reward, episode_steps)
            writer.add_scalar("episode/length", ep_lengths[idx], episode_steps)
            episode_steps += 1

    # Close writer
    writer.close()
