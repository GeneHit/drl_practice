from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base.config import BaseConfig
from practice.base.env_typing import ActTypeC, ObsType
from practice.base.trainer import TrainerBase
from practice.utils.env_utils import extract_episode_data_from_infos
from practice.utils_for_coding.context_utils import ACContext
from practice.utils_for_coding.network_utils import MLP
from practice.utils_for_coding.replay_buffer_utils import Experience, ReplayBuffer


class SACActor(nn.Module):
    """The actor network for SAC."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_sizes: Sequence[int],
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=state_dim,
            output_dim=hidden_sizes[-1],
            hidden_sizes=hidden_sizes[:-1],
            activation=nn.ReLU,
            use_layer_norm=True,
        )
        self.mean_linear = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_linear = nn.Linear(hidden_sizes[-1], action_dim)
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Sample a deterministic action when evaluating.

        Returns:
            action: The deterministic action.
        """
        x = self.net(state)
        mean = self.mean_linear(x)
        action = torch.tanh(mean) * self.max_action
        return action

    def sample(self, state: torch.Tensor) -> torch.Tensor:
        """Sample an action when training.

        Returns:
            action: The sampled action.
        """
        normal = self._get_normal_with_forward(state)
        # reparameterization trick
        z = normal.rsample()
        action = torch.tanh(z)
        # scaled to environment action space
        action_scaled = action * self.max_action
        return action_scaled

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Calculate the log probability of the action.

        Returns:
            log_prob: The log probability of the action.
        """
        normal = self._get_normal_with_forward(state)
        log_prob: torch.Tensor = normal.log_prob(action)
        # adjust for Tanh squashing
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        return log_prob

    def _get_normal_with_forward(self, state: torch.Tensor) -> torch.distributions.Normal:
        """Get the normal distribution with the forward pass.

        Returns:
            normal: The normal distribution.
        """
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        return normal


@dataclass(frozen=True, kw_only=True)
class SACConfig(BaseConfig):
    """The configuration for the TD3 algorithm."""

    total_steps: int
    """The step number to train the policy.

    The total step data is timesteps = total_steps / vector_env_num.
    """
    hidden_sizes: tuple[int, ...]
    """The hidden sizes of the MLP."""
    use_layer_norm: bool
    """Whether to use layer normalization."""
    critic_lr: float
    """The learning rate for the critic."""
    replay_buffer_capacity: int
    """The capacity of the replay buffer."""
    batch_size: int
    """The batch size for training."""
    update_start_step: int
    """The step to start updating the critic and actor."""
    alpha: float
    """The temperature parameter for the entropy regularization.

    Balance the exploration and exploitation.
    """
    alpha_lr: float
    """The learning rate for the temperature parameter (auto-tuned)."""
    target_entropy: float
    """The target entropy for the entropy regularization."""
    tau: float
    """The soft update factor for the target networks."""
    max_action: float
    """The maximum action value."""
    log_std_min: float
    """The minimum log standard deviation."""
    log_std_max: float
    """The maximum log standard deviation."""


class SACTrainer(TrainerBase):
    """The trainer for the SAC algorithm."""

    def __init__(self, config: SACConfig, ctx: ACContext) -> None:
        super().__init__(config, ctx)

        self._config: SACConfig = config
        self._ctx: ACContext = ctx

    def train(self) -> None:
        """Train the SAC algorithm.

        Steps:
        1. initialization: replay buffer, _SACPod, reset
        2. loop:
            - interact with environment
            - collect valid data
            - update if > update_start_step:
                - sample batch
                - update actor, critic, alpha, and target networks
        """
        # 1. initializations
        # Initialize tensorboard writer
        writer = SummaryWriter(
            log_dir=Path(self._config.artifact_config.output_dir) / "tensorboard"
        )
        # Use environment from context - must be vector environment
        envs = self._ctx.continuous_envs

        # Create trainer pod
        pod = _SACPod(config=self._config, ctx=self._ctx, writer=writer)
        # Create replay buffer using observation shape from context
        obs_dtype = envs.single_observation_space.dtype
        assert obs_dtype in (np.float32, np.uint8)
        assert envs.single_action_space.dtype == np.float32
        replay_buffer = ReplayBuffer(
            capacity=self._config.replay_buffer_capacity,
            state_shape=self._ctx.env_state_shape,
            # use cast for mypy
            state_dtype=cast(type[np.float32] | type[np.uint8], obs_dtype),
            action_dtype=np.float32,
            action_shape=envs.single_action_space.shape,  # correct for continuous actions
        )

        # Initialize environments
        states, _ = envs.reset()
        assert isinstance(states, np.ndarray), "States must be numpy array"
        # Track previous step terminal status to avoid invalid transitions
        prev_dones: NDArray[np.bool_] = np.zeros(envs.num_envs, dtype=np.bool_)
        episode_steps = 0

        # 2. loop
        timestep = self._config.total_steps // envs.num_envs
        start_step = self._config.update_start_step // envs.num_envs
        for step in tqdm(range(timestep), desc="Training"):
            # Get actions for all environments
            actions = pod.action(states, step)
            # Step the environment
            next_states, rewards, terminated, truncated, infos = envs.step(actions)

            # Cast rewards to numpy array for indexing
            rewards = np.asarray(rewards, dtype=np.float32)
            # Handle terminal observations and create proper training transitions
            dones = np.logical_or(terminated, truncated, dtype=np.bool_)

            # Only store transitions for states that were not terminal in the previous step
            # we use AutoReset wrapper, so the envs will be reset automatically when it's done
            # when any done in n step, the next_states of n+1 step is the first of the next episode
            pre_non_terminal_mask = ~prev_dones
            if np.any(pre_non_terminal_mask):
                # Create training transitions
                replay_buffer.add_batch(
                    states=states[pre_non_terminal_mask],
                    actions=actions[pre_non_terminal_mask],
                    rewards=rewards[pre_non_terminal_mask],
                    next_states=next_states[pre_non_terminal_mask],
                    dones=dones[pre_non_terminal_mask],
                )

            states = next_states
            prev_dones = dones

            # Training updates
            if step >= start_step:
                if len(replay_buffer) < self._config.batch_size:
                    continue
                pod.update(replay_buffer.sample(self._config.batch_size))

            # Log episode metrics
            ep_rewards, ep_lengths = extract_episode_data_from_infos(infos)
            for idx, reward in enumerate(ep_rewards):
                writer.add_scalar("episode/reward", reward, episode_steps)
                writer.add_scalar("episode/length", ep_lengths[idx], episode_steps)
                episode_steps += 1


class _SACPod:
    """The SAC pod for training."""

    def __init__(self, config: SACConfig, ctx: ACContext, writer: SummaryWriter) -> None:
        self._config: SACConfig = config
        self._ctx: ACContext = ctx
        self._writer: SummaryWriter = writer

        assert isinstance(self._ctx.network, SACActor)
        self._actor = self._ctx.network

    def action(self, state: NDArray[ObsType], step: int) -> NDArray[ActTypeC]:
        """Sample an action from the actor network.

        Returns:
            action: The sampled action for training.
        """
        with torch.no_grad():
            a = self._actor.sample(torch.from_numpy(state).float().to(self._config.device))
        return a.cpu().numpy()

    def update(self, experience: Experience) -> None:
        """Update the actor network."""
        raise NotImplementedError("Not implemented")
