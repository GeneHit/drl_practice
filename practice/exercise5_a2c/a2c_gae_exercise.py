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
from practice.utils.env_utils import extract_episode_data_from_infos
from practice.utils_for_coding.network_utils import init_weights
from practice.utils_for_coding.scheduler_utils import ScheduleBase


class ActorCritic(nn.Module):
    """The actor-critic network for the A2C algorithm."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int) -> None:
        super().__init__()
        # shared feedforward network
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # actor head: output logits for each action
        self.policy_logits = nn.Linear(hidden_size, n_actions)
        # critic head: output value V(s)
        self.value_head = nn.Sequential(nn.Linear(hidden_size, 1))

        # initialize parameters, it is optional but helps to stabilize the training
        self.shared_layers.apply(init_weights)
        init_weights(self.policy_logits)
        self.value_head.apply(init_weights)

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

    total_steps: int
    """The sum of step of all environments to get the data for training the policy.

    The update number is total_steps / (rollout_len * vector_env_num).
    """

    rollout_len: int
    """The length of each rollout.

    The rollout is a sequence of states, actions, rewards, values, log_probs, entropies, dones.
    """

    gae_lambda_or_n_step: float | int
    """The GAE lambda or TD(n) step.

    If the value is a float, it is the GAE lambda.
    If the value is an integer, it is the TD(n) step.
    """

    entropy_coef: ScheduleBase
    """The entropy coefficient for the entropy loss."""

    value_loss_coef: float = 0.5
    """The coefficient for the value loss."""

    grad_acc: int = 1
    """The number of rollouts to accumulate gradients."""

    max_grad_norm: float | None = None
    """The maximum gradient norm for gradient clipping."""

    critic_lr: float
    """The learning rate for the critic."""

    critic_lr_gamma: float | None = None
    """The gamma for the critic learning rate scheduler."""


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
        1. Reset
        2. Run one rollout and buffer the data
        3. Update the policy/value network
        4. Reset the buffer
        5. go to 2, until the total steps is reached
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
        episode_num = 0
        assert self._config.env_config.vector_env_num is not None
        num_updates = self._config.total_steps // (
            self._config.rollout_len * self._config.env_config.vector_env_num
        )
        state, _ = envs.reset()
        for rollout_idx in tqdm(range(num_updates), desc="Rollouts"):
            # 2. Run one rollout
            for _ in range(self._config.rollout_len):
                # sample action and buffer partial data
                actions = pod.action(state)

                # step environment
                # we use AutoReset wrapper, so the envs will be reset automatically when it's done
                # when any done in n step, the next_states of n+1 step is the first of the next episode
                next_states, rewards, terms, truncs, infos = envs.step(actions)
                dones = np.logical_or(terms, truncs)

                # buffer the data after stepping the environment
                # when any pre done, it will buffer a bad transition between two episodes.
                pod.add_stepped_data(
                    next_states=next_states, rewards=rewards.astype(np.float32), dones=dones
                )

                # update state
                state = next_states

                # record the episode data
                episode_rewards, episode_lengths = extract_episode_data_from_infos(infos)
                episode_num += len(episode_rewards)
                if episode_rewards:
                    writer.add_scalar("episode/reward", np.mean(episode_rewards), episode_num)
                    writer.add_scalar("episode/length", np.mean(episode_lengths), episode_num)

            # 3. Update the policy/value network
            pod.update()

            # 4. Clear the buffer
            pod.reset(rollout_idx=rollout_idx)

        writer.close()


@dataclass(kw_only=True, frozen=True)
class _StepData:
    """One-step data of multi environments."""

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

    The buffer is used to store the rollout data (Sequence of _RolloutStep) of multi environments:
    - _RolloutStep: states, actions, rewards, values, log_probs, entropies, dones

    Because of the gymnasium's AutoReset wrapper, the rollout data may be discontinuous, which means
    the rollout contains multiple episodes's data.
    Bad transition between two episodes, for one environment:
        - if it's done in n-step, the next_states of n+1 step is the first of the next episode
        - so, transition (s_n, a_n, r_n, s_n+1) is a bad data. Have to be ignored.

    Example:
        1. s_n-1, a_n-1, r_n-1, s_n     (last step of episode 1)
        2. s_n, a_n, r_n, s_n+1         (bad transition between two episodes)
        3. s_n+1, a_n+1, r_n+1, s_n+2   (first step of episode 2)
    """

    def __init__(self) -> None:
        self._rollout: list[_StepData] = []
        self._temp_data: _StepData | None = None

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
        self._temp_data = _StepData(
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

    def get_data(self) -> list[_StepData]:
        """Get the data from the buffer."""
        return self._rollout

    def clear(self, writer: SummaryWriter | None = None, rollout_idx: int = 0) -> None:
        """Clear the buffer."""
        if writer is not None:
            self._record_scalars(writer, rollout_idx)

        self._rollout.clear()
        self._temp_data = None

    def __len__(self) -> int:
        return len(self._rollout)

    def _record_scalars(self, writer: SummaryWriter, rollout_idx: int) -> None:
        """Record the scalars to the writer."""
        # TODO: record more scalars
        pass


class _A2CPod(abc.ABC):
    """The pod base for the A2C algorithm."""

    def __init__(self, config: A2CConfig, ctx: ContextBase, writer: SummaryWriter) -> None:
        self._config: A2CConfig = config
        self._ctx: ContextBase = ctx
        self._writer: SummaryWriter = writer
        self._rollout: _RolloutBuffer = _RolloutBuffer()

    def reset(self, rollout_idx: int | None = None) -> None:
        """Reset the pod."""
        if rollout_idx is not None:
            self._rollout.clear(writer=self._writer, rollout_idx=rollout_idx)

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
        self._rollout_count: int = 0
        self._pre_last_dones: Tensor | None = None

    def update(self) -> None:
        """Update the actor and critic."""
        # 1. Get the data from the buffer
        rollout = self._rollout.get_data()

        # 2. Compute the advantages [rollout_len, N_env]
        advantages, values, rewards, entropies = self._compute_advantages_and_filter(rollout)

        # 3. Compute the loss
        # the actor/policy loss
        log_probs = torch.stack([step.log_probs for step in rollout])  # [rollout_len, N_env]
        # normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss = -(advantages * log_probs).mean()
        # the critic/value loss
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        value_mse = F.mse_loss(values, rewards)
        value_loss = self._config.value_loss_coef * value_mse
        # the entropy loss
        entropy_coef = self._config.entropy_coef(self._rollout_count)
        entropy = entropies.mean()
        entropy_loss = -entropy_coef * entropy
        # the total loss
        total_loss = pg_loss + value_loss + entropy_loss

        # 4. Update the actor and critic
        total_loss.backward()
        if self._config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._ctx.network.parameters(), self._config.max_grad_norm
            )
        self._ctx.optimizer.step()
        self._ctx.step_lr_schedulers()
        self._ctx.optimizer.zero_grad()

        # 5. Log the loss
        self._writer.add_scalar("other/value_mse", value_mse.item(), self._rollout_count)
        self._writer.add_scalar("other/entropy", entropy.item(), self._rollout_count)
        self._writer.add_scalar("other/entropy_coef", entropy_coef, self._rollout_count)

        self._writer.add_scalar("losses/policy_loss", pg_loss.item(), self._rollout_count)
        self._writer.add_scalar("losses/value_loss", value_loss.item(), self._rollout_count)
        self._writer.add_scalar("losses/entropy_loss", entropy_loss.item(), self._rollout_count)
        self._writer.add_scalar("losses/total_loss", total_loss.item(), self._rollout_count)
        self._rollout_count += 1

    def _compute_advantages_and_filter(
        self, rollout: list[_StepData]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the advantages and filter the bad transition between two episodes.

        Returns:
            advantages: Tensor, shape [T, N]
                the valid advantages of the rollout
            values: Tensor, shape [T, N]
                the valid values of the rollout
            rewards: Tensor, shape [T, N]
                the valid rewards of the rollout
            entropies: Tensor, shape [T, N]
                the valid entropies of the rollout
        """
        t_1_data = rollout[-1]
        assert t_1_data.values.ndim == 1
        # total steps T, where rollout timestamp is 0, 1, ..., T-1
        t = len(rollout)
        device = self._config.device
        gamma = self._config.gamma
        assert isinstance(self._config.gae_lambda_or_n_step, float), (
            "Use float for GAE, int for TD(n)."
        )
        gae_lambda: float = self._config.gae_lambda_or_n_step

        # get the value of the last step V_T:
        # 1) get the V(s_T) of all envs
        states_tensor = _np2tensor(t_1_data.states.astype(np.float32), device)
        _, v_all = self._ctx.network(states_tensor)
        v_all = v_all.view(-1)
        # 2) mask the done envs: done→0, not done→V(s_T)
        done = torch.from_numpy(t_1_data.dones).to(device)
        v_t = v_all * (~done)

        # stack the data to [T, N]
        rewards = torch.stack([_np2tensor(step.rewards, device) for step in rollout])
        dones = torch.stack([_np2tensor(step.dones.astype(np.float32), device) for step in rollout])
        values = torch.stack([step.values for step in rollout])

        # construct the next values
        next_values = torch.empty_like(values)
        next_values[t - 1] = v_t
        next_values[:-1] = values[1:]

        advantages = torch.empty_like(values)
        gae = torch.zeros_like(values[0])
        # compute the advantages in reverse order
        for t_step in range(t - 1, -1, -1):
            # the mask will cut off the
            mask = 1 - dones[t_step]
            delta = rewards[t_step] + gamma * next_values[t_step] - values[t_step]
            gae = delta + gamma * gae_lambda * gae * mask
            advantages[t_step] = gae

        # filter the bad transition between two episodes
        advantages, values, rewards, entropies = self._filter_bad_transition(
            dones=torch.stack([torch.from_numpy(step.dones).to(device) for step in rollout]),
            advantages=advantages,
            entropies=torch.stack([step.entropies for step in rollout]),
            values=values,
            rewards=rewards,
        )

        return advantages, values, rewards, entropies

    def _filter_bad_transition(
        self, dones: Tensor, advantages: Tensor, entropies: Tensor, values: Tensor, rewards: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Filter the bad transition between two episodes.

        Because pg_loss = -log_probs * advantages, so not need to filter the log_probs.

        See _RolloutBuffer for the reason and more details.

        Args:
            dones: Tensor, shape [T, N]
            advantages: Tensor, shape [T, N]
            entropies: Tensor, shape [T, N]
            values: Tensor, shape [T, N]
        """
        # filter the bad transition between two episodes: pre_dones == 1
        prev_dones = torch.empty_like(dones, dtype=torch.bool)
        if self._pre_last_dones is None:
            self._pre_last_dones = torch.zeros_like(dones[-1], dtype=torch.bool)
        assert self._pre_last_dones is not None  # make mypy happy
        prev_dones[0] = self._pre_last_dones
        prev_dones[1:] = dones[:-1]
        self._pre_last_dones = prev_dones[-1]
        # get the valid mask [T, N]
        valid_mask = ~prev_dones

        # filter the data
        advantages = advantages * valid_mask
        entropies = entropies * valid_mask
        values = values * valid_mask
        rewards = rewards * valid_mask

        return advantages, values, rewards, entropies


def _np2tensor(x: NDArray[np.float32], device: torch.device) -> Tensor:
    """Convert a numpy array to a tensor."""
    return torch.from_numpy(x).to(device)
