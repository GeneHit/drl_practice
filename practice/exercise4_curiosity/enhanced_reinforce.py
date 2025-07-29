from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base.chest import RewardBase, RewardConfig
from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ObsType
from practice.base.trainer import TrainerBase
from practice.utils_for_coding.baseline_utils import (
    BaselineBase,
    ConstantBaseline,
)
from practice.utils_for_coding.scheduler_utils import ScheduleBase
from practice.utils_for_coding.writer_utils import log_stats


@dataclass(kw_only=True, frozen=True)
class EnhancedReinforceConfig(BaseConfig):
    total_steps: int
    """The total number of steps to train the policy."""

    lr_gamma: float
    """The learning rate gamma for the optimizer."""

    hidden_sizes: tuple[int, ...]
    """The hidden sizes for the policy network."""

    baseline: BaselineBase = ConstantBaseline()
    """The baseline for the policy.

    The baseline is used to reduce the variance of the policy gradient.
    """

    entropy_coef: ScheduleBase
    """The entropy coefficient for the entropy loss.

    The entropy loss is added to the policy loss to encourage the policy to explore the environment.
    """

    reward_configs: tuple[RewardConfig, ...] = ()
    """The reward configs for the rewarders."""


@dataclass(kw_only=True, frozen=True)
class ReinforceContext(ContextBase):
    rewarders: tuple[RewardBase, ...] = ()


class EnhancedReinforceTrainer(TrainerBase):
    """A trainer for the enhanced reinforce algorithm.

    This trainer is used to train the policy network with a single environment.
    """

    def __init__(self, config: EnhancedReinforceConfig, ctx: ReinforceContext) -> None:
        super().__init__(config=config, ctx=ctx)
        self._config: EnhancedReinforceConfig = config
        self._ctx: ReinforceContext = ctx

    def train(self) -> None:
        """Train the policy network with a single environment."""
        # Initialize tensorboard writer
        writer = SummaryWriter(
            log_dir=Path(self._config.artifact_config.output_dir) / "tensorboard"
        )
        env = self._ctx.env
        # Create training pod and buffer
        pod = _EnhancedReinforcePod(config=self._config, ctx=self._ctx, writer=writer)
        episode_buffer = _EpisodeBuffer()

        step = 0
        step_bar = tqdm(total=self._config.total_steps, desc="Training")
        while step < self._config.total_steps:
            # 1. Resets for new episode
            state, _ = env.reset()
            done = False

            # 2. Run one episode
            while not done:
                # Sample action
                action, log_prob, entropy = pod.action_and_log_prob(state.reshape(1, -1))

                # Step environment
                next_state, reward, term, trunc, _ = env.step(action)
                done = bool(term or trunc)

                # Store data in episode buffer
                episode_buffer.add(float(reward), log_prob, entropy)
                # Calculate intrinsic rewards
                for rewarder in self._ctx.rewarders:
                    intrinsic_rewards = rewarder.get_reward(
                        state.reshape(1, -1), next_state.reshape(1, -1), step
                    )
                    assert intrinsic_rewards[0] < 30, f"{intrinsic_rewards=}"
                    episode_buffer.add_intrinsic_reward(
                        intrinsic_reward=float(intrinsic_rewards[0]),
                        rewarder_name=rewarder.__class__.__name__,
                    )

                # Update state
                state = next_state
                step += 1
                step_bar.update(1)

            # 3. Process completed episode
            # Update policy with the episode data
            rewards, log_probs, entropies = episode_buffer.get_episode_data()
            pod.update(rewards, log_probs, entropies, step)
            # Clear episode buffer and log rewarder data if present
            episode_buffer.clear(writer, step)

        # Close writer
        writer.close()


class _EpisodeBuffer:
    """Simple buffer to store episode data for REINFORCE."""

    def __init__(self) -> None:
        self._log_probs: list[Tensor] = []
        self._entropies: list[Tensor] = []
        # the original rewards
        self._rewards: list[float] = []
        # the intrinsic rewards
        self._intrinsic_rewards: list[float] = []
        # the contributions of each rewarder
        self._rewarder_contributions: dict[str, list[float]] = {}
        self._episode_count = 0

    def add(self, reward: float, log_prob: Tensor, entropy: Tensor) -> None:
        """Add a step to the episode buffer.

        Args:
            state: The state of the step.
            action: The action of the step.
            reward: The original reward of the step.
        """
        self._rewards.append(reward)
        self._log_probs.append(log_prob)
        self._entropies.append(entropy)
        self._intrinsic_rewards.append(0.0)  # Initialize with 0
        # Initialize rewarder contributions for this step
        for rewarder_name in self._rewarder_contributions:
            self._rewarder_contributions[rewarder_name].append(0.0)

    def add_intrinsic_reward(self, intrinsic_reward: float, rewarder_name: str = "unknown") -> None:
        """Add intrinsic reward to the last step. Should be called after add()."""
        if len(self._intrinsic_rewards) == 0:
            raise RuntimeError("No intrinsic rewards to add, please call add() first.")

        self._intrinsic_rewards[-1] += intrinsic_reward
        # Track individual rewarder contribution
        if rewarder_name not in self._rewarder_contributions:
            # Initialize this rewarder's history with zeros for previous steps
            self._rewarder_contributions[rewarder_name] = [0.0] * len(self._intrinsic_rewards)
        # Ensure the contribution list is the same length as intrinsic_rewards
        while len(self._rewarder_contributions[rewarder_name]) < len(self._intrinsic_rewards):
            self._rewarder_contributions[rewarder_name].append(0.0)
        self._rewarder_contributions[rewarder_name][-1] += intrinsic_reward

    def get_episode_data(
        self,
    ) -> tuple[list[float], list[Tensor], list[Tensor]]:
        return (
            # self._rewards .+ self._intrinsic_rewards,
            [a + b for a, b in zip(self._rewards, self._intrinsic_rewards)],
            self._log_probs,
            self._entropies,
        )

    def clear(self, writer: SummaryWriter | None = None, episodes_completed: int = 0) -> None:
        """Clear the episode buffer and optionally log rewarder data.

        Args:
            writer: Optional SummaryWriter for logging rewarder data
            episodes_completed: Episode count for logging
        """
        # Log rewarder data before clearing if writer is provided
        if writer is not None:
            self._record_scalars(writer, episodes_completed)

        # Clear all data
        self._log_probs.clear()
        self._entropies.clear()
        self._rewards.clear()
        self._intrinsic_rewards.clear()
        self._rewarder_contributions.clear()

    def __len__(self) -> int:
        return len(self._rewards)

    def _record_scalars(self, writer: SummaryWriter, episodes_completed: int) -> None:
        """Private method to record all rewarder-related scalars to tensorboard."""
        writer.add_scalar("episode/length", len(self._rewards), episodes_completed)
        writer.add_scalar("episode/reward", sum(self._rewards), episodes_completed)
        if len(self._rewarder_contributions) > 0:
            # Log intrinsic and total rewards
            writer.add_scalar(
                "episode/intrinsic_reward", sum(self._intrinsic_rewards), episodes_completed
            )
            writer.add_scalar(
                "episode/total_reward",
                sum(self._rewards) + sum(self._intrinsic_rewards),
                episodes_completed,
            )

            # Log individual rewarder contributions
            for rewarder_name in self._rewarder_contributions:
                writer.add_scalar(
                    f"episode/{rewarder_name}",
                    sum(self._rewarder_contributions[rewarder_name]),
                    episodes_completed,
                )
        self._episode_count += 1


class _EnhancedReinforcePod:
    def __init__(
        self,
        config: EnhancedReinforceConfig,
        ctx: ReinforceContext,
        writer: SummaryWriter,
    ) -> None:
        self._config = config
        self._ctx = ctx
        self._writer = writer

        # Initialize gradients
        self._ctx.optimizer.zero_grad()
        # Ensure network is in training mode
        self._ctx.network.train()

    def action_and_log_prob(
        self, state: NDArray[ObsType], actions: Sequence[ActType] | None = None
    ) -> tuple[ActType, Tensor, Tensor]:
        """Get action(s), log_prob(s), and entropy for state(s).

        Args:
            state: Single state or batch of states
            actions: If provided, compute log probs for these actions. If None, sample new actions.

        Returns:
            actions: Tensor
                If actions was None: newly sampled actions
                If actions was provided: the same actions as a tensor
            log_probs: Tensor
                Log probabilities for the actions
            entropies: Tensor
                Entropy values for the policy distribution
        """
        is_single = len(state.shape) == len(self._ctx.env_state_shape)
        state_batch = state if not is_single else state.reshape(1, *state.shape)
        state_tensor = torch.from_numpy(state_batch).to(self._config.device)

        probs = self._ctx.network(state_tensor)
        dist = torch.distributions.Categorical(probs)
        entropies = dist.entropy()

        if actions is None:
            # Sample new actions
            sampled_actions = dist.sample()
            log_probs = dist.log_prob(sampled_actions)
            return ActType(sampled_actions.cpu().item()), log_probs, entropies
        else:
            # Compute log probs for given actions
            action_tensor = torch.tensor(actions, device=self._config.device)
            log_probs = dist.log_prob(action_tensor)
            return ActType(action_tensor.cpu().item()), log_probs, entropies

    def update(
        self,
        rewards: Sequence[float],
        log_probs: Sequence[Tensor],
        entropies: Sequence[Tensor],
        step: int,
    ) -> None:
        """Update the policy network with a episode's data.

        Don't support inputting batch episodes now, because it's complex
        to collect the multi-episode batch data:
        1. for single env, use gradient accumulation is a better way.
        2. for multi env, different env has different episode length.
            - so it's complex to collect the multi-episode batch data.
            - if use the same episode length, it's not efficient.

        Gradient accumulation is useful for:
        1. Single env: accumulate multiple episodes in serial.
        2. Multi env: accumulate episodes from parallel environments.

        Parameters
        ----------
            rewards: Sequence[float]
                rewards of a episode, [episode_length, ]
            log_probs: Tensor
                log probabilities of actions in a episode, [episode_length, ]
            entropies: Tensor
                entropy values for the policy distribution, [episode_length, ]
        """
        # Calculate returns (discounted cumulative rewards)
        returns: list[float] = [0.0] * len(rewards)
        disc_return_t = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            disc_return_t = rewards[i] + self._config.gamma * disc_return_t
            returns[i] = disc_return_t

        # Convert to tensor
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        # Update baseline
        baseline_values = self._config.baseline.update(returns, log_probs)
        baseline_tensor = torch.tensor(baseline_values, dtype=torch.float32)
        # [episode_length, ] -> [episode_length, 1]
        advantages = (returns_tensor - baseline_tensor).reshape(-1, 1)

        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate policy loss and entropy loss
        log_probs_tensor = torch.stack(tuple(log_probs)).reshape(-1, 1)
        entropies_tensor = torch.stack(tuple(entropies))
        pg_loss = -(log_probs_tensor * advantages.to(self._config.device)).mean()
        entropy_coef = self._config.entropy_coef(step)
        entropy = entropies_tensor.mean()
        entropy_loss = -entropy_coef * entropy
        total_loss = pg_loss + entropy_loss

        # compute gradients and update the network
        self._ctx.optimizer.zero_grad()
        total_loss.backward()
        if self._config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._ctx.network.parameters(), self._config.max_grad_norm
            )
        self._ctx.optimizer.step()

        # Log training metrics
        log_stats(
            data={
                "loss/policy": pg_loss,
                "loss/entropy": entropy_loss,
                "loss/total": total_loss,
                "other/baseline": baseline_values[0],
                "other/entropy_coef": entropy_coef,
                "other/entropy": entropy,
                "other/advantages": advantages.mean(),
            },
            writer=self._writer,
            step=step,
            log_interval=self._config.log_interval,
        )
