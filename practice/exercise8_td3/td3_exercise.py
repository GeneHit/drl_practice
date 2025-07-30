import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from tqdm import tqdm

from practice.base.config import BaseConfig
from practice.base.env_typing import ActTypeC, ObsType
from practice.base.trainer import TrainerBase
from practice.utils_for_coding.context_utils import ACContext
from practice.utils_for_coding.network_utils import MLP, soft_update
from practice.utils_for_coding.numpy_tensor_utils import as_tensor_on
from practice.utils_for_coding.replay_buffer_utils import Experience, ReplayBuffer
from practice.utils_for_coding.scheduler_utils import ScheduleBase
from practice.utils_for_coding.writer_utils import CustomWriter


class TD3Actor(nn.Module):
    """The actor network for TD3.

    The output is scaled to [-1,1] by `max_action`, for continuous action space.
    """

    def __init__(
        self, state_dim: int, action_dim: int, max_action: float, hidden_sizes: Sequence[int]
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=nn.ReLU,
            output_activation=nn.Tanh,  # for scaling output to [-1,1]
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(state) * self.max_action)


@dataclass(frozen=True, kw_only=True)
class TD3Config(BaseConfig):
    """The configuration for the TD3 algorithm."""

    total_steps: int
    """The step number to train the policy.

    The total step data is timesteps = total_steps / vector_env_num.
    """
    hidden_sizes: tuple[int, ...]
    """The hidden sizes of the MLP."""
    critic_lr: float
    """The learning rate for the critic."""
    replay_buffer_capacity: int
    """The capacity of the replay buffer."""
    batch_size: int
    """The batch size for training."""
    update_start_step: int
    """The step to start updating the critic and actor."""
    policy_delay: int
    """The step interval for updating the policy.

    The policy is updated every `policy_delay` critic updates.
    """
    policy_noise: float = 0.2
    """The standard deviation of the noise for the policy. Default Gaussian noise."""
    noise_clip: float = 0.5
    """The clip value of the noise for the policy."""
    exploration_noise: ScheduleBase
    """The exploration noise for the policy."""
    max_action: float
    """The maximum action value."""
    tau: float = 0.005
    """The soft update factor for the target networks."""


class TD3Trainer(TrainerBase):
    """The trainer for the TD3 algorithm."""

    def __init__(self, config: TD3Config, ctx: ACContext) -> None:
        super().__init__(config, ctx)

        self._config: TD3Config = config
        self._ctx: ACContext = ctx

    def train(self) -> None:
        """Train the TD3 algorithm.

        Steps:
        1. initialization: replay buffer, _TD3Pod, reset
        2. loop:
            - interact with environment
            - collect valid data
            - update if > update_start_step:
                - sample batch
                - update critic
                - update actor/target_network if necessary
            - log episode data if has
        """
        # 1. initializations
        # Initialize tensorboard writer
        writer = CustomWriter(
            track=self._ctx.track_and_evaluate,
            log_dir=self._config.artifact_config.get_tensorboard_dir(),
        )
        # Use environment from context - must be vector environment
        envs = self._ctx.continuous_envs

        # Create trainer pod
        pod = _TD3Pod(config=self._config, ctx=self._ctx, writer=writer)
        # Create replay buffer using observation shape from context
        obs_dtype = envs.single_observation_space.dtype
        assert obs_dtype in (np.float32, np.uint8)
        replay_buffer = ReplayBuffer(
            capacity=self._config.replay_buffer_capacity,
            state_shape=self._ctx.env_state_shape,
            # use cast for mypy
            state_dtype=cast(type[np.float32] | type[np.uint8], obs_dtype),
            action_dtype=ActTypeC,
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
                # Only store transitions where the previous step didn't end an episode
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
                # sample batch and update
                pod.update(replay_buffer.sample(self._config.batch_size), step)

            # Log episode metrics
            episode_steps += writer.log_episode_stats_if_has(infos, episode_steps)

        writer.close()


class _TD3Pod:
    """The TD3 pod for training."""

    def __init__(self, config: TD3Config, ctx: ACContext, writer: CustomWriter) -> None:
        self._config: TD3Config = config
        self._ctx: ACContext = ctx
        self._writer: CustomWriter = writer

        self._target_actor = copy.deepcopy(self._ctx.network)
        self._target_critic = copy.deepcopy(self._ctx.critic)
        self._target_actor.eval()
        self._target_critic.eval()

        # convert to tensor on device， avoiding unnecessary device transfers
        self._tau = torch.tensor(self._config.tau, dtype=torch.float32, device=self._config.device)
        self._gamma = as_tensor_on(self._config.gamma, self._tau)
        self._max_action = as_tensor_on(self._config.max_action, self._tau)
        self._policy_noise = as_tensor_on(self._config.policy_noise, self._tau)
        self._noise_clip = as_tensor_on(self._config.noise_clip, self._tau)
        self._one = as_tensor_on(1, self._tau)

    def action(self, state: NDArray[ObsType], step: int) -> NDArray[ActTypeC]:
        """Get the action for the given state.

        Parameters
        ----------
        state : NDArray[ObsType]
        """
        state_tensor = torch.from_numpy(state).to(self._config.device)
        with torch.no_grad():
            action = self._ctx.network(state_tensor).cpu()

        noise_std = self._config.exploration_noise(step)
        noise = torch.randn_like(action) * noise_std
        noised_action = torch.clamp(
            action + noise, -self._config.max_action, self._config.max_action
        )

        # log
        self._writer.log_action_stats(
            actions=action,
            data={"action/noise_std": noise_std},
            step=step,
            log_interval=self._config.log_interval,
            blocked=False,
        )

        # Ensure action shape is (num_envs, action_dim)
        return cast(NDArray[ActTypeC], noised_action.numpy())

    def update(self, experience: Experience, step: int) -> None:
        """Update the TD3 pod.

        Steps:
        1. update critic
        2. update actor/target_network if necessary
        3. log the stats if necessary in background
        """
        exp = experience.to(self._config.device)
        # 1. update critic
        # 1.1 get target Q-value
        target_q = self._get_target_q(exp)
        # 1.2 get critic Q-value
        q = self._ctx.critic(exp.states, exp.actions)
        # Return Sequence: standard TD3, use Q1/Q2, usually train together
        # Return Single: the standard DDPG, use one Q.
        qs = q if isinstance(q, Sequence) else (q,)
        # 1.3 compute critic loss
        critic_loss = torch.stack([nn.functional.mse_loss(qi, target_q) for qi in qs]).mean()
        # 1.4 update critic
        self._ctx.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self._config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self._ctx.critic.parameters(), self._config.max_grad_norm)
        self._ctx.critic_optimizer.step()

        # 2. update actor/target_network if necessary
        actor_data = {}
        if step % self._config.policy_delay == 0:
            # calculate actor loss and update actor
            action = self._ctx.network(exp.states)
            q_value = self._ctx.critic(exp.states, action)
            if isinstance(q_value, Sequence):
                q_value = q_value[0]
            actor_loss = -q_value.mean()
            self._ctx.optimizer.zero_grad()
            actor_loss.backward()
            if self._config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self._ctx.network.parameters(), self._config.max_grad_norm)
            self._ctx.optimizer.step()

            # update target networks (soft update)
            soft_update(self._ctx.network, self._target_actor, self._tau)
            soft_update(self._ctx.critic, self._target_critic, self._tau)
            actor_data["loss/actor"] = actor_loss

        # log the stats if necessary in background
        data = {
            "loss/critic": critic_loss,
            "q_value/target": target_q,
            **{f"q_value/critic_{i}": qi for i, qi in enumerate(qs)},
            **actor_data,
        }
        self._writer.log_stats(
            data=data,
            step=step,
            log_interval=self._config.log_interval,
            blocked=False,
        )

    def _get_target_q(self, experience: Experience) -> torch.Tensor:
        """Get the target Q-value for the given experience.

        Steps:
        1. get next action (with noise) from target actor
        2. get next Q-value from target critic
        3. calculate target Q-value with rewards and next Q-value

        Parameters
        ----------
        experience : Experience
        """
        with torch.no_grad():
            # next action (with noise) from target actor: [batch, action_dim]
            target_action = self._target_actor(experience.next_states)
            target_action_noise = torch.clamp(
                torch.randn_like(target_action) * self._policy_noise,
                -self._noise_clip,
                self._noise_clip,
            )
            next_action = torch.clamp(
                target_action + target_action_noise, -self._max_action, self._max_action
            )

            # next Q-value from target critic
            target_critic_q = self._target_critic(experience.next_states, next_action)

            # Return Sequence: standard TD3, use Q1/Q2, usually train together
            # Return Single: the standard DDPG, use one Q.
            if isinstance(target_critic_q, Sequence):
                next_q: torch.Tensor = torch.min(*target_critic_q)
            else:
                next_q = target_critic_q
            # next_q shape [batch, 1]，rewards shape [batch]
            if next_q.dim() == 2 and next_q.size(1) == 1:
                next_q = next_q.squeeze(-1)

            # calculate target Q-value
            target_q = (
                experience.rewards + self._gamma * (self._one - experience.dones.float()) * next_q
            ).view(-1, 1)
        return target_q
