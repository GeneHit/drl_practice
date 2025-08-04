import abc
from dataclasses import dataclass, replace

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ActTypeC, ObsType
from practice.base.trainer import TrainerBase
from practice.utils_for_coding.network_utils import init_weights
from practice.utils_for_coding.numpy_tensor_utils import argmax_action
from practice.utils_for_coding.scheduler_utils import ScheduleBase
from practice.utils_for_coding.writer_utils import CustomWriter


class ActorCritic(nn.Module):
    """The actor-critic network for the PPO algorithm.

    For the discrete action space.
    """

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

    def action(self, x: Tensor) -> ActType:
        """Get the action for evaluation/gameplay with 1 environment.

        Returns:
            action: The single action.
        """
        logits, _ = self.forward(x)
        return argmax_action(logits, dtype=ActType)


@dataclass(kw_only=True, frozen=True)
class PPOConfig(BaseConfig):
    """The configuration for the PPO algorithm."""

    total_steps: int
    """The sum of step of all environments to get the data for training the policy.

    The update number is total_steps / (rollout_len * vector_env_num).
    """

    rollout_len: int
    """The length of each rollout.

    The rollout is a sequence of states, actions, rewards, values, log_probs, dones.
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

    max_grad_norm: float | None = None
    """The maximum gradient norm for gradient clipping."""

    critic_lr: float
    """The learning rate for the critic."""

    critic_lr_gamma: float | None = None
    """The gamma for the critic learning rate scheduler."""

    hidden_size: int
    """The hidden size for the actor-critic network."""

    num_epochs: int
    """The number of epochs to update the policy."""

    minibatch_num: int
    """The number of the minibatch."""

    clip_coef: float
    """The clip coefficient for the PPO."""


class PPOTrainer(TrainerBase):
    """The trainer for the PPO algorithm.

    Required: the actor-critic network, which means the actor and critic networks are shared.

    The actor-critic network is a shared network that outputs the logits for each action and
    the value of the state. This is a usual practice in the actor-critic family of algorithms.

    In some cases, the actor and critic networks are not shared, even separated by 2 different
    networks.
    """

    def __init__(self, config: PPOConfig, ctx: ContextBase) -> None:
        super().__init__(config=config, ctx=ctx)
        self._config: PPOConfig = config  # make mypy happy

    def train(self) -> None:
        """Train the policy network with a vectorized environment.

        The PPO algorithm is implemented as follows:
        1. Reset
        2. Run one rollout and buffer the data
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
            pod: _PPOPod = _GAEPod(config=self._config, ctx=self._ctx, writer=writer)
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
                episode_num += writer.log_episode_stats_if_has(infos, episode_num)

            # 3. Update the policy/value network
            pod.update()
            # 4. Clear the buffer
            pod.reset()

        writer.close()


@dataclass(kw_only=True, frozen=True)
class _StepData:
    """One-step data of multi environments."""

    states: NDArray[ObsType]
    actions: NDArray[ActType | ActTypeC]
    log_probs: Tensor
    values: Tensor
    next_states: NDArray[ObsType]
    rewards: NDArray[np.float32]
    dones: NDArray[np.bool_]


class _RolloutBuffer:
    """The rollout buffer for the PPO algorithm.

    The buffer is used to store the rollout data (Sequence of _RolloutStep) of multi environments:
    - _RolloutStep: states, actions, rewards, values, log_probs, dones

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
        actions: NDArray[ActType | ActTypeC],
        log_probs: Tensor,
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


class _PPOPod(abc.ABC):
    """The pod base for the PPO algorithm."""

    def __init__(self, config: PPOConfig, ctx: ContextBase, writer: CustomWriter) -> None:
        self._config: PPOConfig = config
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
        states_tensor = torch.from_numpy(states).float().to(self._config.device)
        actor_critic = self._ctx.network
        with torch.no_grad():
            logits, value = actor_critic(states_tensor)

            # get the action, log_prob, entropy
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        actions_np: NDArray[ActType] = actions.cpu().numpy().astype(ActType)

        # buffer the data
        self._rollout.add_before_acting(states, actions_np, log_probs, value)
        return actions_np

    @abc.abstractmethod
    def update(self) -> None:
        """Update the actor and critic."""


class _TDNPod(_PPOPod):
    """The pod for the TD(n) PPO algorithm."""

    def __init__(self, config: PPOConfig, ctx: ContextBase, writer: CustomWriter) -> None:
        super().__init__(config=config, ctx=ctx, writer=writer)

    def update(self) -> None:
        """Update the actor and critic.

        Can skip TD(n) here, because the exercise doesn't use TD(n) default.
        """
        raise NotImplementedError("TD(n) PPO is not implemented yet.")


class _GAEPod(_PPOPod):
    """The pod for the PPO algorithm."""

    def __init__(self, config: PPOConfig, ctx: ContextBase, writer: CustomWriter) -> None:
        super().__init__(config=config, ctx=ctx, writer=writer)
        self._rollout_count: int = 0
        self._pre_last_dones: Tensor | None = None

    def update(self) -> None:
        """Update the actor and critic."""
        # 1. Get the data from the buffer
        rollout = self._rollout.get_data()

        # 2. Compute the advantages [d, ] (d: the valid data length)
        advantages, log_probs, returns, states, actions = self._compute_advantages_and_filter(
            rollout
        )
        # normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Update the policy and value network
        config = self._config
        batch_size = advantages.shape[0]
        minibatch_size = batch_size // config.minibatch_num

        assert config.num_epochs > 0, "The number of epochs must be greater than 0."
        for epoch in range(config.num_epochs):
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = indices[start : start + minibatch_size]

                # 3.1 Update the policy network
                logits, values_pred = self._ctx.network(states[mb_inds])
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs_new = dist.log_prob(actions[mb_inds])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs_new - log_probs[mb_inds])
                unclipped_pg_loss = ratio * advantages[mb_inds]
                clipped_pg_loss = (
                    torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
                    * advantages[mb_inds]
                )
                pg_loss = -torch.mean(torch.min(unclipped_pg_loss, clipped_pg_loss))

                # 3.2 Update the value network
                value_mse = F.mse_loss(values_pred.view(-1), returns[mb_inds])
                value_loss = config.value_loss_coef * value_mse

                entropy_coef = config.entropy_coef(self._rollout_count)
                entropy_loss = -entropy_coef * entropy

                # 3.3 Update the total loss
                total_loss = pg_loss + value_loss + entropy_loss

                # 3.4 Backward and update the parameters
                self._ctx.optimizer.zero_grad()
                total_loss.backward()
                if config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._ctx.network.parameters(), config.max_grad_norm
                    )
                self._ctx.optimizer.step()
                self._ctx.step_lr_schedulers()

        # 4. Log only the loss of the last minibatch for simplicity
        self._writer.log_stats(
            data={
                "loss/policy": pg_loss,
                "loss/value": value_loss,
                "loss/entropy": entropy_loss,
                "loss/total": total_loss,
                "other/value_mse": value_mse,
                "other/entropy": entropy,
                "other/entropy_coef": entropy_coef,
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
            rewards: Tensor, shape [d, ]
                the valid rewards of the rollout
            states: Tensor, shape [d, obs_dim]
                the valid states of the rollout
            actions: Tensor, shape [d, ]
                the valid actions of the rollout
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
        with torch.no_grad():
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
            returns=returns,
            states=torch.stack(
                [_np2tensor(step.states.astype(np.float32), device) for step in rollout]
            ),
            actions=torch.stack(
                [_np2tensor(step.actions.astype(np.float32), device) for step in rollout]
            ),
        )

    def _filter_bad_transition(
        self,
        dones: Tensor,
        log_probs: Tensor,
        advantages: Tensor,
        returns: Tensor,
        states: Tensor,
        actions: Tensor,
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
            returns: Tensor, shape [T, N]
            states: Tensor, shape [T, N]
            actions: Tensor, shape [T, N]
        Returns
        -------
        The valid data of the rollout (d (<= T * N) is the valid data length):
            advantages: Tensor, shape [d, ]
            log_probs: Tensor, shape [d, ]
            returns: Tensor, shape [d, ]
            states: Tensor, shape [d, obs_dim]
            actions: Tensor, shape [d, ]
        """
        # filter the bad transition between two episodes: pre_dones == 1
        prev_dones = torch.empty_like(dones, dtype=torch.bool)
        if self._pre_last_dones is None:
            self._pre_last_dones = torch.zeros_like(dones[-1], dtype=torch.bool)
        assert self._pre_last_dones is not None  # make mypy happy
        prev_dones[0] = self._pre_last_dones
        prev_dones[1:] = dones[:-1]
        self._pre_last_dones = dones[-1]
        # get the valid mask [T, N]
        valid_mask = ~prev_dones

        # filter out the invalid data: [T, N] -> [d, ]
        mask_flat = valid_mask.view(-1)
        advantages = advantages.view(-1)[mask_flat]
        log_probs = log_probs.view(-1)[mask_flat]
        returns = returns.view(-1)[mask_flat]
        # dims of states is [T, N, obs_dim] -> [d, obs_dim]
        states = states.view(-1, states.shape[-1])[mask_flat]
        actions = actions.view(-1)[mask_flat]

        return advantages, log_probs, returns, states, actions


def _np2tensor(x: NDArray[np.float32], device: torch.device) -> Tensor:
    """Convert a numpy array to a tensor."""
    return torch.from_numpy(x).to(device)
