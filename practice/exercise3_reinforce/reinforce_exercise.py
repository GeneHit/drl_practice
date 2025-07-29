from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ObsType
from practice.base.trainer import TrainerBase
from practice.utils_for_coding.network_utils import MLP
from practice.utils_for_coding.scheduler_utils import ScheduleBase
from practice.utils_for_coding.writer_utils import log_stats


class Reinforce1DNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=nn.ReLU,
        )
        self._softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = self._softmax(self.mlp(x))  # make mypy happy
        return y


@dataclass(kw_only=True, frozen=True)
class ReinforceConfig(BaseConfig):
    """Configuration for vanilla REINFORCE algorithm."""

    episode: int
    """The number of episodes to train the policy."""

    entropy_coef: ScheduleBase
    """The entropy coefficient for the entropy loss.

    The entropy loss is added to the policy loss to encourage the policy to explore the environment.
    """

    hidden_sizes: tuple[int, ...]
    """The hidden sizes for the policy network."""


class ReinforceTrainer(TrainerBase):
    """A trainer for the vanilla REINFORCE algorithm."""

    def __init__(self, config: ReinforceConfig, ctx: ContextBase) -> None:
        super().__init__(config=config, ctx=ctx)
        self._config: ReinforceConfig = config
        self._ctx: ContextBase = ctx

    def train(self) -> None:
        """Train the policy network with a single environment."""
        # Initialize tensorboard writer
        writer = SummaryWriter(log_dir=self._config.artifact_config.get_tensorboard_dir())
        env = self._ctx.env

        # Create training pod and buffer
        pod = _ReinforcePod(config=self._config, ctx=self._ctx, writer=writer)
        episode_buffer = _EpisodeBuffer()

        for episode_idx in tqdm(range(self._config.episode), desc="Training"):
            # 1. Resets for new episode
            state, _ = env.reset()
            done = False

            # 2. Run one episode
            while not done:
                # Sample action
                action, log_prob, entropy = pod.action_and_log_prob(state)
                # Step environment
                next_state, reward, term, trunc, _ = env.step(action)

                # Store data in episode buffer
                episode_buffer.add(float(reward), log_prob, entropy)
                # Update state
                state = next_state
                done = bool(term or trunc)

            # 3. Process completed episode
            # Update policy with the episode data
            rewards, log_probs, entropies = episode_buffer.get_episode_data()
            pod.update(rewards, log_probs, entropies)

            # Clear episode buffer and log episode data if present
            episode_buffer.clear(writer, episode_idx)

        # Close writer
        writer.close()


class _EpisodeBuffer:
    """Simple buffer to store episode data for vanilla REINFORCE."""

    def __init__(self) -> None:
        self._log_probs: list[Tensor] = []
        self._entropies: list[Tensor] = []
        self._rewards: list[float] = []

    def add(self, reward: float, log_prob: Tensor, entropy: Tensor) -> None:
        """Add a step to the episode buffer.

        Args:
            reward: The reward of the step.
            log_prob: The log probability of the action.
            entropy: The entropy of the policy distribution.
        """
        self._rewards.append(reward)
        self._log_probs.append(log_prob)
        self._entropies.append(entropy)

    def get_episode_data(
        self,
    ) -> tuple[list[float], list[Tensor], list[Tensor]]:
        """Get the episode data.

        Returns:
            tuple containing rewards, log_probs, and entropies
        """
        return self._rewards, self._log_probs, self._entropies

    def clear(self, writer: SummaryWriter | None = None, episodes_completed: int = 0) -> None:
        """Clear the episode buffer and optionally log episode data.

        Args:
            writer: Optional SummaryWriter for logging episode data
            episodes_completed: Episode count for logging
        """
        # Log episode data before clearing if writer is provided
        if writer is not None:
            self._record_scalars(writer, episodes_completed)

        # Clear all data
        self._log_probs.clear()
        self._entropies.clear()
        self._rewards.clear()

    def __len__(self) -> int:
        return len(self._rewards)

    def _record_scalars(self, writer: SummaryWriter, episodes_completed: int) -> None:
        """Private method to record episode scalars to tensorboard."""
        writer.add_scalar("episode/length", len(self._rewards), episodes_completed)
        writer.add_scalar("episode/reward", sum(self._rewards), episodes_completed)


class _ReinforcePod:
    """Internal pod for REINFORCE training logic."""

    def __init__(
        self,
        config: ReinforceConfig,
        ctx: ContextBase,
        writer: SummaryWriter,
    ) -> None:
        self._config = config
        self._ctx = ctx
        self._writer = writer

        # Initialize gradients
        self._ctx.optimizer.zero_grad()
        # Ensure network is in training mode
        self._ctx.network.train()

        self._episode_count = 0
        # Track episodes since last optimizer step
        self._accumulated_episodes = 0

    def action_and_log_prob(
        self, state: NDArray[ObsType], actions: Sequence[ActType] | None = None
    ) -> tuple[ActType, Tensor, Tensor]:
        """Get action(s), log_prob(s), and entropy for state(s).

        Args:
            state: Single state or batch of states
            actions: If provided, compute log probs for these actions. If None, sample new actions.

        Returns:
            action: Sampled action (scalar)
            log_prob: Log probability for the action
            entropy: Entropy value for the policy distribution
        """
        is_single = len(state.shape) == len(self._ctx.env_state_shape)
        state_batch = state if not is_single else state.reshape(1, *state.shape)
        state_tensor = torch.from_numpy(state_batch).to(self._config.device)

        probs = self._ctx.network(state_tensor)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy()

        if actions is None:
            # Sample new actions
            sampled_action = dist.sample()
            log_prob = dist.log_prob(sampled_action)
            # Return scalar action for single state
            if is_single:
                return ActType(sampled_action.item()), log_prob.squeeze(), entropy.squeeze()
            else:
                # For batch, return numpy array of actions
                batch_actions = sampled_action.cpu().numpy().astype(np.int64)
                return batch_actions[0], log_prob, entropy
        else:
            # Compute log probs for given actions
            action_tensor = torch.tensor(actions, device=self._config.device)
            log_prob = dist.log_prob(action_tensor)
            return ActType(action_tensor.item()), log_prob, entropy

    def update(
        self, rewards: Sequence[float], log_probs: Sequence[Tensor], entropies: Sequence[Tensor]
    ) -> None:
        """Update the policy network with an episode's data.

        Args:
            rewards: Rewards from the episode
            log_probs: Log probabilities from the episode
            entropies: Entropies from the episode
        """
        # Calculate returns (discounted cumulative rewards)
        returns = []
        disc_return_t = 0.0
        for reward in reversed(rewards):
            disc_return_t = reward + self._config.gamma * disc_return_t
            returns.append(disc_return_t)
        returns.reverse()

        # Convert to tensor: [episode_length, ] -> [episode_length, 1]
        returns_ts = torch.tensor(returns, dtype=torch.float32).reshape(-1, 1)
        # Normalize advantages for stability
        if len(returns_ts) > 1:
            returns_ts = (returns_ts - returns_ts.mean()) / (returns_ts.std() + 1e-8)

        # Calculate policy loss and entropy loss
        log_probs_tensor = torch.stack(tuple(log_probs)).reshape(-1, 1)
        entropies_tensor = torch.stack(tuple(entropies))

        pg_loss = -(log_probs_tensor * returns_ts.to(self._config.device)).mean()
        entropy_coef = self._config.entropy_coef(self._episode_count)
        entropy = entropies_tensor.mean()
        entropy_loss = -entropy_coef * entropy
        total_loss = pg_loss + entropy_loss

        # Accumulate gradients
        total_loss.backward()
        self._ctx.optimizer.step()
        self._ctx.optimizer.zero_grad()

        # Log training metrics
        log_stats(
            data={
                "loss/policy": pg_loss,
                "loss/entropy": entropy_loss,
                "loss/total": total_loss,
                "entropy/coef": entropy_coef,
                "entropy/value": entropy,
            },
            writer=self._writer,
            step=self._episode_count,
        )

        self._episode_count += 1
