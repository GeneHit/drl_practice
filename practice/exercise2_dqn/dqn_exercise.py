import copy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Discrete
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ArrayType, ObsType
from practice.base.trainer import TrainerBase
from practice.utils.env_utils import extract_episode_data_from_infos
from practice.utils_for_coding.replay_buffer_utils import Experience, ReplayBuffer
from practice.utils_for_coding.scheduler_utils import LinearSchedule

# Type aliases for vector environments
EnvsType = gym.vector.VectorEnv[NDArray[ObsType], ActType, NDArray[ArrayType]]


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


@dataclass(kw_only=True, frozen=True)
class DQNConfig(BaseConfig):
    """Configuration for DQN algorithm."""

    timesteps: int
    start_epsilon: float = 1.0
    end_epsilon: float = 0.01
    exploration_fraction: float = 0.1
    replay_buffer_capacity: int = 10000
    batch_size: int = 32
    train_interval: int = 1
    target_update_interval: int = 100
    update_start_step: int = 100


class DQNTrainer(TrainerBase):
    """A trainer for the DQN algorithm."""

    def __init__(self, config: DQNConfig, ctx: ContextBase) -> None:
        super().__init__(config=config, ctx=ctx)
        self._config: DQNConfig = config
        self._ctx: ContextBase = ctx

    def train(self) -> None:
        """Train the DQN agent with multiple environments."""
        # Initialize tensorboard writer
        writer = SummaryWriter(
            log_dir=Path(self._config.artifact_config.output_dir) / "tensorboard"
        )

        # Use environment from context - must be vector environment for DQN training
        envs = self._ctx.envs

        # Create trainer pod
        pod = _DQNPod(config=self._config, ctx=self._ctx, writer=writer)

        # Create replay buffer using observation shape from context
        obs_shape = self._ctx.env_state_shape
        replay_buffer = ReplayBuffer(
            capacity=self._config.replay_buffer_capacity,
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

        # Training loop
        pbar = tqdm(total=self._config.timesteps, desc="Training")
        for step in range(self._config.timesteps):
            # Get actions for all environments
            actions = pod.action(states, step)

            # Step all environments
            next_states, rewards, terms, truncs, infos = envs.step(actions.tolist())

            # Cast rewards to numpy array for indexing
            rewards = np.asarray(rewards, dtype=np.float32)

            # Handle terminal observations and create proper training transitions
            dones = np.logical_or(terms, truncs, dtype=np.bool_)

            # Only store transitions for states that were not terminal in the previous step
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
            if step >= self._config.update_start_step:
                if step % self._config.train_interval == 0:
                    experiences = replay_buffer.sample(self._config.batch_size)
                    loss = pod.update(experiences)
                    writer.add_scalar("training/loss", loss, step)

                if step % self._config.target_update_interval == 0:
                    pod.sync_target_net()

            # Log episode metrics
            ep_rewards, ep_lengths = extract_episode_data_from_infos(infos)
            for idx, reward in enumerate(ep_rewards):
                writer.add_scalar("episode/reward", reward, episode_steps)
                writer.add_scalar("episode/length", ep_lengths[idx], episode_steps)
                episode_steps += 1

            pbar.update(1)

        # Cleanup
        pbar.close()
        writer.close()


class _DQNPod:
    """Internal pod for DQN training logic."""

    def __init__(
        self,
        config: DQNConfig,
        ctx: ContextBase,
        writer: SummaryWriter,
    ) -> None:
        self._config = config
        self._ctx = ctx
        self._writer = writer

        # Create target network
        self._target_net = copy.deepcopy(ctx.network)
        self._target_net.eval()

        # Create epsilon schedule
        self._epsilon_schedule = LinearSchedule(
            start_e=config.start_epsilon,
            end_e=config.end_epsilon,
            duration=int(config.exploration_fraction * config.timesteps),
        )

        # Get action space info
        assert isinstance(self._ctx.eval_env.action_space, Discrete)
        self._action_n = int(self._ctx.eval_env.action_space.n)

    def sync_target_net(self) -> None:
        """Synchronize target network with current Q-network."""
        self._target_net.load_state_dict(self._ctx.network.state_dict())

    def action(self, state: NDArray[ObsType], step: int, eval: bool = False) -> NDArray[ActType]:
        """Get action(s) for state(s).

        Args:
            state: Single state or batch of states
            step: Current step in the training process
            eval: If True, use greedy policy (epsilon=0)

        Returns:
            actions: NDArray[ActType]
                Single action or batch of actions depending on input shape.
        """
        # Check if input is a single state or batch of states
        is_single = len(state.shape) == len(self._ctx.env_state_shape)
        state_batch = state if not is_single else state.reshape(1, *state.shape)

        if eval:
            # Evaluation phase: always greedy
            state_tensor = torch.from_numpy(state_batch).to(self._config.device)
            with torch.no_grad():
                q_values = self._ctx.network(state_tensor).cpu()
                actions = q_values.argmax(dim=1).numpy().astype(ActType)
            return cast(NDArray[ActType], actions)

        # Training phase: epsilon-greedy
        batch_size = state_batch.shape[0]
        actions = np.zeros(batch_size, dtype=ActType)

        # Random mask for exploration
        epsilon = self._epsilon_schedule(step)
        random_mask = np.random.random(batch_size) < epsilon
        # Random actions for exploration
        num_random = int(np.sum(random_mask))
        actions[random_mask] = np.random.randint(0, self._action_n, size=num_random, dtype=ActType)

        # Greedy actions for exploitation
        if not np.all(random_mask):
            exploit_states = state_batch[~random_mask]
            state_tensor = torch.from_numpy(exploit_states).to(self._config.device)
            self._ctx.network.train()
            with torch.no_grad():
                q_values = self._ctx.network(state_tensor).cpu()
                greedy_actions = q_values.argmax(dim=1).numpy().astype(ActType)
                actions[~random_mask] = greedy_actions

        # Log epsilon
        self._writer.add_scalar("training/epsilon", epsilon, step)

        return cast(NDArray[ActType], actions)

    def update(self, experiences: Experience) -> float:
        """Update Q-network using experiences.

        Args:
            experiences: Batch of experiences from replay buffer

        Returns:
            loss: The TD loss value for logging
        """
        # Move all inputs to device
        states = experiences.states.to(self._config.device)
        actions = experiences.actions.view(-1, 1).to(self._config.device)
        rewards = experiences.rewards.to(self._config.device)
        next_states = experiences.next_states.to(self._config.device)
        dones = experiences.dones.to(self._config.device)

        # Compute TD target using target network
        with torch.no_grad():
            target_max: Tensor = self._target_net(next_states).max(dim=1)[0]
            td_target = rewards.flatten() + self._config.gamma * target_max * (
                1 - dones.flatten().float()
            )

        # Get current Q-values for the actions taken
        self._ctx.network.train()
        current_q = self._ctx.network(states).gather(1, actions).squeeze()

        # Compute loss and update
        self._ctx.optimizer.zero_grad()
        loss = F.mse_loss(current_q, td_target)
        loss.backward()
        self._ctx.optimizer.step()

        return float(loss.item())
