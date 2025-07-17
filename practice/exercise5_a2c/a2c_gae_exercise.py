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
from practice.utils_for_coding.network_utils import init_weights


class ActorCritic(nn.Module):
    """The actor-critic network for the A2C algorithm."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int) -> None:
        super().__init__()
        # shared feedforward network
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # actor head: output logits for each action
        self.policy_logits = nn.Linear(hidden_size, n_actions)
        # critic head: output value V(s)
        self.value_head = nn.Linear(hidden_size, 1)

        # initialize parameters, it is optional but helps to stabilize the training
        self.shared_layers.apply(init_weights)
        for layer in [self.policy_logits, self.value_head]:
            init_weights(layer)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """The forward pass of the actor-critic network.

        Args:
            x: tensor, shape [batch_size, obs_dim]

        Returns:
            logits: tensor, shape [batch_size, n_actions]
            value: tensor, shape [batch_size]
        """
        x = self.shared_layers(x)

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
                # sample action and buffer partial data
                actions = pod.action(state)

                # step environment
                next_states, rewards, terms, truncs, _ = envs.step(actions)
                dones = np.logical_or(terms, truncs)

                # buffer the data after stepping the environment
                pod.add_stepped_data(
                    next_states=next_states, rewards=rewards.astype(np.float32), dones=dones
                )

                # update state
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
            # use empty to avoid initialization, it will be filled in the add_after_acting
            next_states=np.empty(states.shape, dtype=np.float32),
            rewards=np.empty(len(states), dtype=np.float32),
            dones=np.empty(len(states), dtype=np.bool_),
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
        # get policy logits and value
        states_tensor = torch.from_numpy(states).float().to(self._config.device)
        actor_critic = self._ctx.network
        logits, value = actor_critic(states_tensor)

        # get the action, log_prob, entropy
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropies = dist.entropy()
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        actions_np: NDArray[ActType] = actions.cpu().numpy().astype(ActType)

        # buffer the data
        self._rollout.add_before_acting(states, actions_np, log_probs, entropies, value)
        return actions_np

    @abc.abstractmethod
    def update(self) -> None:
        """Update the actor and critic."""


class _TDNPod(_A2CPod):
    """The pod for the TD(n) A2C algorithm."""

    def __init__(self, config: A2CConfig, ctx: ContextBase, writer: SummaryWriter) -> None:
        super().__init__(config=config, ctx=ctx, writer=writer)

    def update(self) -> None:
        """Update the actor and critic."""
        raise NotImplementedError("A2C is not implemented yet.")


class _GAEPod(_A2CPod):
    """The pod for the A2C-GAE algorithm."""

    def __init__(self, config: A2CConfig, ctx: ContextBase, writer: SummaryWriter) -> None:
        super().__init__(config=config, ctx=ctx, writer=writer)
        self._episode_count: int = 0

    def update(self) -> None:
        """Update the actor and critic."""
        # 1. Get the data from the buffer
        rollout = self._rollout.get_data()

        # 2. Compute the advantages
        advantages = self._compute_advantages(rollout)

        # 3. Compute the loss
        # the actor/policy loss
        log_probs = torch.stack([step.log_probs for step in rollout])
        pg_loss = -(advantages * log_probs).mean()
        # the critic/value loss
        values = torch.stack([step.values for step in rollout])
        rewards = torch.stack([_np2tensor(step.rewards, self._config.device) for step in rollout])
        value_loss = F.mse_loss(values, rewards)
        # the entropy loss
        entropy_loss = (
            -self._config.entropy_coef * torch.stack([step.entropies for step in rollout]).mean()
        )
        # the total loss
        total_loss = pg_loss + value_loss + entropy_loss

        # 4. Update the actor and critic
        self._ctx.optimizer.zero_grad()
        total_loss.backward()
        self._ctx.optimizer.step()

        # 5. Log the loss
        self._writer.add_scalar("losses/policy_loss", pg_loss.item(), self._episode_count)
        self._writer.add_scalar("losses/value_loss", value_loss.item(), self._episode_count)
        self._writer.add_scalar("losses/entropy_loss", entropy_loss.item(), self._episode_count)
        self._writer.add_scalar("losses/total_loss", total_loss.item(), self._episode_count)
        self._episode_count += 1

    def _compute_advantages(self, rollout: list[_RolloutStep]) -> Tensor:
        """Compute the advantages."""
        t_1_data = rollout[-1]
        assert t_1_data.values.ndim == 1
        # total steps T, where rollout timestamp is 0, 1, ..., T-1
        t = len(rollout)
        envs_num = len(t_1_data.values)
        device = self._config.device
        gamma = self._config.gamma
        assert isinstance(self._config.gae_lambda_or_n_step, float), (
            "Use float for GAE, int for TD(n)."
        )
        gae_lambda: float = self._config.gae_lambda_or_n_step

        # get the value of the last step V_T:
        #   if the episode is done, the advantage is 0
        #   if the episode is not done, the advantage is the V(s_T)
        v_t = torch.zeros((envs_num,), dtype=torch.float32, device=device)
        not_done_idx = np.where(~t_1_data.dones)[0]
        not_done_s_t = t_1_data.states[not_done_idx]
        _, not_done_v_t = self._ctx.network(torch.from_numpy(not_done_s_t).to(device))
        v_t[not_done_idx] = not_done_v_t

        # td error
        td_error = torch.empty((t, envs_num), dtype=torch.float32, device=device)
        # advantage
        a_t = torch.empty((t, envs_num), dtype=torch.float32, device=device)
        # compute the td error and advantage
        td_error[t - 1] = _np2tensor(t_1_data.rewards, device) + gamma * v_t - t_1_data.values
        a_t[t - 1] = td_error[t - 1]
        # form T-2 to 0
        for t_step in range(t - 2, -1, -1):
            step_data = rollout[t_step]
            td_error[t_step] = (
                _np2tensor(step_data.rewards, device)
                + gamma * td_error[t_step + 1]
                - step_data.values
            )
            a_t[t_step] = td_error[t_step] + gamma * gae_lambda * a_t[t_step + 1]

        return a_t


def _np2tensor(x: NDArray[np.float32], device: torch.device) -> Tensor:
    """Convert a numpy array to a tensor."""
    return torch.from_numpy(x).to(device)
