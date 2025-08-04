import abc
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ObsType
from practice.base.trainer import TrainerBase
from practice.utils_for_coding.network_utils import MLP, init_weights
from practice.utils_for_coding.numpy_tensor_utils import argmax_action
from practice.utils_for_coding.scheduler_utils import ScheduleBase
from practice.utils_for_coding.writer_utils import CustomWriter


class ActorCritic(nn.Module):
    """The actor-critic network for the A2C algorithm."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        # shared feedforward network
        self.shared_layers = MLP(
            input_dim=obs_dim,
            output_dim=hidden_sizes[-1],
            hidden_sizes=hidden_sizes[:-1],
            activation=nn.ReLU,
            use_layer_norm=True,
        )

        # actor head: output logits for each action
        self.policy_logits = nn.Linear(hidden_sizes[-1], n_actions)
        # critic head: output value V(s)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

        # initialize parameters, it is optional but helps to stabilize the training
        init_weights(self.policy_logits)
        init_weights(self.value_head)

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

    def action(self, x: Tensor) -> ActType:
        """Get the action for evaluation/gameplay with 1 environment.

        Returns:
            action: The single action.
        """
        logits, _ = self.forward(x)
        return argmax_action(logits, dtype=ActType)


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

    critic_lr: float
    """The learning rate for the critic."""

    critic_lr_gamma: float | None = None
    """The gamma for the critic learning rate scheduler."""

    normalize_returns: bool = False
    """Whether to normalize the returns."""

    hidden_sizes: tuple[int, ...]
    """The hidden sizes for the actor-critic network."""

    # for the high-variance reward.
    reward_clip: tuple[float, float] | None = None
    """The clip range for the reward before calculating the advantages.

    If tuple, use the clip range [-clip_min, clip_max].
    If None, don't clip the reward.
    """
    value_clip_range: float | None = None
    """The clip range for the value loss.

    The clip will work during backward, does nothing during forward.
    """


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
        2. Run one rollout:
            a. interact with the environment
            b. buffer the data
            c. log the episode data if has
        3. Update the policy/value network
        4. Reset the buffer
        5. go to 2, until the total steps is reached
        """
        writer = CustomWriter(
            track=self._ctx.track_and_evaluate,
            log_dir=self._config.artifact_config.get_tensorboard_dir(),
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
        rollout_num = self._config.total_steps // (
            self._config.rollout_len * self._config.env_config.vector_env_num
        )
        state, _ = envs.reset()

        for _ in tqdm(range(rollout_num), desc="Rollouts"):
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
                episode_num += writer.log_episode_stats_if_has(
                    infos, episode_num, log_interval=self._config.log_interval
                )

            # 3. Update the policy/value network
            pod.update()
            # 4. Clear the buffer
            pod.reset()

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

    def clear(self) -> None:
        """Clear the buffer."""
        self._rollout.clear()
        self._temp_data = None

    def __len__(self) -> int:
        return len(self._rollout)


class _A2CPod(abc.ABC):
    """The pod base for the A2C algorithm."""

    def __init__(self, config: A2CConfig, ctx: ContextBase, writer: CustomWriter) -> None:
        self._config: A2CConfig = config
        self._ctx: ContextBase = ctx
        self._writer: CustomWriter = writer
        self._rollout: _RolloutBuffer = _RolloutBuffer()

    def reset(self) -> None:
        """Reset the pod."""
        self._rollout.clear()

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
        states_tensor = _np2tensor(states.astype(np.float32), self._config.device)
        logits, value = self._ctx.network(states_tensor)

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

    def __init__(self, config: A2CConfig, ctx: ContextBase, writer: CustomWriter) -> None:
        super().__init__(config=config, ctx=ctx, writer=writer)

    def update(self) -> None:
        """Update the actor and critic."""
        raise NotImplementedError("A2C is not implemented yet.")


class _GAEPod(_A2CPod):
    """The pod for the A2C-GAE algorithm."""

    def __init__(self, config: A2CConfig, ctx: ContextBase, writer: CustomWriter) -> None:
        super().__init__(config=config, ctx=ctx, writer=writer)
        self._rollout_count: int = 0
        self._pre_last_dones: Tensor | None = None

    def update(self) -> None:
        """Update the actor and critic.

        Steps:
        1. Get the data from the buffer
        2. Compute the advantages [d, ] (d: the valid data length)
        3. Compute the loss
        4. Update the actor and critic
        5. Log the stats if necessary in background
        """
        # 1. Get the data from the buffer
        rollout = self._rollout.get_data()

        # 2. Compute the advantages [d, ] (d: the valid data length)
        advantages, log_probs, values, returns, entropies = self._compute_advantages_and_filter(
            rollout
        )

        # 3. Compute the loss
        # normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss = -(advantages * log_probs).mean()
        # the critic/value loss
        if self._config.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # the value loss
        value_loss = self._value_loss(values, returns)
        coef_value_loss = self._config.value_loss_coef * value_loss
        # the entropy loss
        entropy_coef = self._config.entropy_coef(self._rollout_count)
        entropy = entropies.mean()
        entropy_loss = -entropy_coef * entropy
        # the total loss
        total_loss = pg_loss + coef_value_loss + entropy_loss

        # 4. Update the actor and critic
        total_loss.backward()
        if self._config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._ctx.network.parameters(), self._config.max_grad_norm
            )
        self._ctx.optimizer.step()
        self._ctx.step_lr_schedulers()
        self._ctx.optimizer.zero_grad()

        # 5. Log the stats
        self._writer.log_stats(
            data={
                "loss/policy": pg_loss,
                "loss/value": coef_value_loss,
                "loss/entropy": entropy_loss,
                "loss/total": total_loss,
                "entropy/entropy": entropy,
                "entropy/coef": entropy_coef,
            },
            step=self._rollout_count,
            log_interval=self._config.log_interval,
            blocked=False,
        )
        self._rollout_count += 1

    def _compute_advantages_and_filter(
        self, rollout: list[_StepData]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute the advantages and filter the bad transition between two episodes.

        Returns:
            advantages: Tensor, shape [d, ]
                the valid advantages of the rollout
            log_probs: Tensor, shape [d, ]
                the valid log_probs of the rollout
            values: Tensor, shape [d, ]
                the valid values of the rollout
            rewards: Tensor, shape [d, ]
                the valid rewards of the rollout
            entropies: Tensor, shape [d, ]
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
        if isinstance(self._config.reward_clip, tuple):
            rewards = torch.clamp(rewards, self._config.reward_clip[0], self._config.reward_clip[1])
        dones = torch.stack([_np2tensor(step.dones.astype(np.float32), device) for step in rollout])
        values = torch.stack([step.values for step in rollout])

        # construct the next values
        next_values = torch.empty_like(values)
        next_values[t - 1] = v_t
        next_values[:-1] = values[1:]

        advantages = torch.empty_like(values)
        returns = torch.empty_like(values)
        gae = torch.zeros_like(values[0])

        # compute the advantages in reverse order
        for t_step in range(t - 1, -1, -1):
            # the mask will cut off the
            mask = 1 - dones[t_step]
            delta = rewards[t_step] + gamma * next_values[t_step] - values[t_step]
            gae = delta + gamma * gae_lambda * gae * mask
            advantages[t_step] = gae
            returns[t_step] = gae + values[t_step]

        # filter the bad transition between two episodes: [T, N] -> [d, ]
        return self._filter_bad_transition(
            dones=torch.stack([torch.from_numpy(step.dones).to(device) for step in rollout]),
            log_probs=torch.stack([step.log_probs for step in rollout]),
            advantages=advantages,
            entropies=torch.stack([step.entropies for step in rollout]),
            values=values,
            returns=returns,
        )

    def _filter_bad_transition(
        self,
        dones: Tensor,
        log_probs: Tensor,
        advantages: Tensor,
        entropies: Tensor,
        values: Tensor,
        returns: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Filter the bad transition between two episodes.

        See _RolloutBuffer for the reason and more details.

        Have to throw away the data of the bad transition between two episodes, don't use its data
        and compute graph of backward.

        Parameters
        ----------
            dones: Tensor, shape [T, N]
            log_probs: Tensor, shape [T, N]
            advantages: Tensor, shape [T, N]
            entropies: Tensor, shape [T, N]
            values: Tensor, shape [T, N]
            returns: Tensor, shape [T, N]

        Returns
        -------
        The valid data of the rollout (d (<= T * N) is the valid data length):
            advantages: Tensor, shape [d, ]
            log_probs: Tensor, shape [d, ]
            values: Tensor, shape [d, ]
            returns: Tensor, shape [d, ]
            entropies: Tensor, shape [d, ]
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

        # filter out the invalid data: [T, N] -> [d, ]
        mask_flat = valid_mask.view(-1)
        advantages = advantages.view(-1)[mask_flat]
        log_probs = log_probs.view(-1)[mask_flat]
        entropies = entropies.view(-1)[mask_flat]
        values = values.view(-1)[mask_flat]
        returns = returns.view(-1)[mask_flat]

        return advantages, log_probs, values, returns, entropies

    def _value_loss(self, values: Tensor, returns: Tensor) -> Tensor:
        """Compute the value MSE loss.

        Clip the value if configured.

        Args:
            values: The predicted value.
            returns: The returns.
        """
        if self._config.value_clip_range is None:
            return self._loss(values, returns)

        # Use current prediction as baseline (detach it for clipping)
        values_detached = values.detach()

        # unclipped and clipped losses
        value_mse = self._loss(values, returns, reduction="none")
        # the clamp works during backward, does nothing during forward
        values_clipped = values_detached + torch.clamp(
            values - values_detached,
            -self._config.value_clip_range,
            self._config.value_clip_range,
        )
        value_mse_clipped = self._loss(values_clipped, returns, reduction="none")

        # clip the loss
        loss = 0.5 * torch.max(value_mse, value_mse_clipped).mean()

        self._writer.log_stats(
            data={
                "value_loss/original": value_mse,
                "value_loss/clipped": value_mse_clipped,
                "value_loss/loss": loss,
            },
            step=self._rollout_count,
            log_interval=self._config.log_interval,
            blocked=False,
        )
        return loss

    def _loss(self, values: Tensor, returns: Tensor, reduction: str = "mean") -> Tensor:
        """Compute the MSE or smooth L1 loss.

        Args:
            values: The predicted value.
            returns: The returns.
        """
        return F.mse_loss(input=values.view(-1), target=returns, reduction=reduction)


def _np2tensor(x: NDArray[np.float32], device: torch.device) -> Tensor:
    """Convert a numpy array to a tensor."""
    return torch.from_numpy(x).to(device)
