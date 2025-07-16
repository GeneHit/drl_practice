import abc
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ObsType
from practice.base.trainer import TrainerBase


class ActorCritic(nn.Module):
    """The actor-critic network for the A2C algorithm."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int) -> None:
        super().__init__()
        # shared feedforward network
        self.shared_fc1 = nn.Linear(obs_dim, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)

        # actor head: output logits for each action
        self.policy_logits = nn.Linear(hidden_size, n_actions)
        # critic head: output value V(s)
        self.value_head = nn.Linear(hidden_size, 1)

        # initialize parameters, it is optional but helps to stabilize the training
        for layer in [self.shared_fc1, self.shared_fc2, self.policy_logits, self.value_head]:
            nn.init.orthogonal_(layer.weight, gain=int(nn.init.calculate_gain("relu")))
            nn.init.constant_(layer.bias, 0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """The forward pass of the actor-critic network.

        Args:
            x: tensor, shape [batch_size, obs_dim]

        Returns:
            logits: tensor, shape [batch_size, n_actions]
            value: tensor, shape [batch_size]
        """
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        logits = self.policy_logits(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


@dataclass(kw_only=True, frozen=True)
class A2CConfig(BaseConfig):
    """The configuration for the A2C-GAE algorithm."""

    episode: int
    """The number of episodes to train the policy."""

    rollout_len: int
    """The length of each rollout.

    The rollout is a sequence of states, actions, rewards, values, log_probs, entropies, dones.
    """

    gae_lambda_or_n_step: float | int
    """The GAE lambda or TD(n) step.

    If the value is a float, it is the GAE lambda.
    If the value is an integer, it is the TD(n) step.
    """

    entropy_coef: float = 0.01
    """The entropy coefficient for the entropy loss."""

    grad_acc: int = 1
    """The number of rollouts to accumulate gradients."""


class A2CTrainer(TrainerBase):
    """The trainer for the A2C algorithm.

    Required: the actor-critic network, which means the actor and critic networks are shared.

    The actor-critic network is a shared network that outputs the logits for each action and
    the value of the state. This is a usual practice in the actor-critic family of algorithms.

    In some cases, the actor and critic networks are not shared, even separated by 2 different
    networks.
    """

    def __init__(self, config: A2CConfig, ctx: ContextBase) -> None:
        super().__init__(config=config, ctx=ctx)
        self._config: A2CConfig = config
        self._ctx: ContextBase = ctx

    def train(self) -> None:
        """Train the policy network with a vectorized environment.

        The A2C algorithm is implemented as follows:
        1. Reset for new rollout
        2. Run one rollout and buffer the data
        3. Update the policy/value network
        4. Reset the buffer
        5. Repeat
        """
        writer = SummaryWriter(
            log_dir=Path(self._config.artifact_config.output_dir) / "tensorboard"
        )
        # only support the vectorized environment
        envs = self._ctx.envs
        if isinstance(self._config.gae_lambda_or_n_step, float):
            pod: _A2CPod = _GAEPod(config=self._config, ctx=self._ctx, writer=writer)
        else:
            pod = _TDNPod(config=self._config, ctx=self._ctx, writer=writer)

        # Create variables for loop
        for episode_idx in tqdm(range(self._config.episode), desc="Episodes"):
            # 1. Reset for new rollout
            state, _ = envs.reset()

            # 2. Run one rollout
            for _ in range(self._config.rollout_len):
                # Sample action and buffer partial data
                actions = pod.action(state)

                # Step environment
                next_states, rewards, terms, truncs, _ = envs.step(actions)
                dones = np.logical_or(terms, truncs)
                pod.add_stepped_data(
                    next_states=next_states, rewards=rewards.astype(np.float32), dones=dones
                )

                # Update state
                state = next_states
                if dones.any():
                    break

            # 3. Update the policy/value network
            pod.update()

            # 4. Clear the buffer
            pod.reset(episode_completed=episode_idx)

        writer.close()


@dataclass(kw_only=True, frozen=True)
class _RolloutStep:
    """One-step data of a rollout."""

    states: NDArray[ObsType]
    actions: NDArray[ActType]
    log_probs: Tensor
    entropies: Tensor
    values: Tensor
    next_states: NDArray[ObsType]
    rewards: NDArray[np.float32]
    dones: NDArray[np.bool_]


class _RolloutBuffer:
    """The rollout buffer for the A2C algorithm.

    The buffer is used to store the rollouts:
    - states, actions, rewards, values, log_probs, entropies, dones
    """

    def __init__(self) -> None:
        self._rollout: list[_RolloutStep] = []
        self._temp_data: _RolloutStep | None = None

    def add_before_acting(
        self,
        states: NDArray[ObsType],
        actions: NDArray[ActType],
        log_probs: Tensor,
        entropies: Tensor,
        values: Tensor,
    ) -> None:
        """Add partial rollout to the buffer before acting.

        The data is generated when getting the action for step environment.
        """
        assert self._temp_data is None, "The previous temp data is not cleared."
        self._temp_data = _RolloutStep(
            states=states,
            actions=actions,
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            next_states=np.zeros(states.shape, dtype=np.float32),
            rewards=np.zeros(len(states), dtype=np.float32),
            dones=np.zeros(len(states), dtype=np.bool_),
        )

    def add_after_acting(
        self,
        next_states: NDArray[ObsType],
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
    ) -> None:
        """Add partial rollout to the buffer after acting."""
        assert self._temp_data is not None, "The previous temp data is not added."
        rollout = replace(self._temp_data, next_states=next_states, rewards=rewards, dones=dones)
        self._rollout.append(rollout)
        self._temp_data = None

    def get_data(self) -> list[_RolloutStep]:
        """Get the data from the buffer."""
        return self._rollout

    def clear(self, writer: SummaryWriter | None = None, episode_completed: int = 0) -> None:
        """Clear the buffer."""
        if writer is not None:
            self._record_scalars(writer, episode_completed)

        self._rollout.clear()
        self._temp_data = None

    def __len__(self) -> int:
        return len(self._rollout)

    def _record_scalars(self, writer: SummaryWriter, episodes_completed: int) -> None:
        """Record the scalars to the writer."""
        # TODO: record more scalars
        writer.add_scalar("episode/length", len(self._rollout), episodes_completed)


class _A2CPod(abc.ABC):
    """The pod base for the A2C algorithm."""

    def __init__(self, config: A2CConfig, ctx: ContextBase, writer: SummaryWriter) -> None:
        self._config: A2CConfig = config
        self._ctx: ContextBase = ctx
        self._writer: SummaryWriter = writer
        self._rollout: _RolloutBuffer = _RolloutBuffer()

    def reset(self, episode_completed: int | None = None) -> None:
        """Reset the pod."""
        if episode_completed is not None:
            self._rollout.clear(writer=self._writer, episode_completed=episode_completed)

    def add_stepped_data(
        self,
        next_states: NDArray[ObsType],
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
    ) -> None:
        """Add the data after stepping the environment.

        Args:
            next_states: The next states.
            rewards: The rewards.
            dones: The dones.
        """
        self._rollout.add_after_acting(next_states, rewards, dones)

    def action(self, states: NDArray[ObsType]) -> NDArray[ActType]:
        """Get the actions of the policy and buffer partial data.

        Args:
            states: The states to get the action of.

        Returns:
            actions: The actions.
        """
        states_tensor = torch.from_numpy(states).float().to(self._config.device)
        actor_critic = self._ctx.network
        logits, value = actor_critic(states_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropies = dist.entropy()
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        actions_np: NDArray[ActType] = actions.cpu().numpy().astype(np.int64)
        self._rollout.add_before_acting(states, actions_np, log_probs, entropies, value)
        return actions_np

    @abc.abstractmethod
    def update(self) -> None:
        """Update the policy/value network."""


class _TDNPod(_A2CPod):
    """The pod for the TD(n) A2C algorithm."""

    def __init__(self, config: A2CConfig, ctx: ContextBase, writer: SummaryWriter) -> None:
        super().__init__(config=config, ctx=ctx, writer=writer)

    def action(self, states: NDArray[ObsType]) -> NDArray[ActType]:
        """Get the actions of the policy.

        Args:
            states: The states to get the action of.

        Returns:
            actions: The actions.
        """
        raise NotImplementedError("A2C is not implemented yet.")

    def update(self) -> None:
        """Update the policy/value network."""
        raise NotImplementedError("A2C is not implemented yet.")


class _GAEPod(_A2CPod):
    """The pod for the A2C-GAE algorithm."""

    def __init__(self, config: A2CConfig, ctx: ContextBase, writer: SummaryWriter) -> None:
        super().__init__(config=config, ctx=ctx, writer=writer)

    def update(self) -> None:
        """Update the policy/value network."""
        raise NotImplementedError("A2C-GAE is not implemented yet.")
