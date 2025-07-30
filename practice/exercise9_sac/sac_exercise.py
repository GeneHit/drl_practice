import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from tqdm import tqdm

from practice.base.config import BaseConfig
from practice.base.env_typing import ActTypeC, ObsType
from practice.base.trainer import TrainerBase
from practice.utils_for_coding.context_utils import ACContext
from practice.utils_for_coding.network_utils import MLP, soft_update
from practice.utils_for_coding.numpy_tensor_utils import as_tensor_on
from practice.utils_for_coding.replay_buffer_utils import Experience, ReplayBuffer
from practice.utils_for_coding.writer_utils import CustomWriter


class SACActor(nn.Module):
    """The actor network for SAC."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float,
        action_bias: float,
        hidden_sizes: Sequence[int],
        log_std_min: float,
        log_std_max: float,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=state_dim,
            output_dim=hidden_sizes[-1],
            hidden_sizes=hidden_sizes[:-1],
            activation=nn.ReLU,
            use_layer_norm=use_layer_norm,
        )
        self.mean_linear = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_linear = nn.Linear(hidden_sizes[-1], action_dim)

        # for type checking
        self.action_scale: torch.Tensor
        self.action_bias: torch.Tensor
        self.log_std_min: torch.Tensor
        self.log_std_max: torch.Tensor
        # register the parameters
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("log_std_min", torch.tensor(log_std_min, dtype=torch.float32))
        self.register_buffer("log_std_max", torch.tensor(log_std_max, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Sample a deterministic action when evaluating.

        Returns:
            action: The deterministic action.
        """
        x = self.net(state)
        mean = self.mean_linear(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action when training.

        Returns:
            action: The sampled action.
            log_prob: The log probability of the action.
        """
        # 1. get mean and log_std
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        #  map log_std to [log_std_min, log_std_max] for stability
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (1 + log_std)

        # 2. get normal distribution with mean and std
        normal = torch.distributions.Normal(mean, log_std.exp())

        # 3. reparameterization trick (mean + std * N(0,1))
        z = normal.rsample()

        # 4. scale to environment action space
        action = torch.tanh(z)
        action_scaled = action * self.action_scale + self.action_bias

        # 5. calculate log_prob
        log_prob: torch.Tensor = normal.log_prob(z) - torch.log(
            self.action_scale * (1 - action.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # [batch, 1]
        return action_scaled, log_prob


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
    auto_tune_alpha: bool
    """Whether to auto-tune the temperature parameter."""
    alpha_lr: float
    """The learning rate for the temperature parameter (auto-tuned)."""
    target_entropy: float
    """The target entropy for the entropy regularization."""
    tau: float
    """The soft update factor for the target networks."""
    max_action: float
    """The maximum action value."""
    log_std_min: float = -20
    """The minimum log standard deviation."""
    log_std_max: float = 2
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
            - log episode metrics
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
                pod.update(replay_buffer.sample(self._config.batch_size), step)

            # Log episode metrics
            episode_steps += writer.log_episode_stats_if_has(infos, episode_steps)

        writer.close()


class _SACPod:
    """The SAC pod for training."""

    def __init__(self, config: SACConfig, ctx: ACContext, writer: CustomWriter) -> None:
        self._config: SACConfig = config
        self._ctx: ACContext = ctx
        self._writer: CustomWriter = writer
        assert isinstance(self._ctx.network, SACActor)
        self._actor = self._ctx.network
        self._actor.train()

        self._target_critic = copy.deepcopy(self._ctx.critic)
        self._target_critic.eval()
        self._ctx.critic.train()
        self._log_alpha = torch.tensor(
            np.log(config.alpha), dtype=torch.float32, device=config.device, requires_grad=True
        )
        self._alpha_optimizer = torch.optim.Adam([self._log_alpha], lr=config.alpha_lr)

        # convert to tensor on device， avoiding unnecessary device transfers
        self._tau = torch.tensor(self._config.tau, dtype=torch.float32, device=self._config.device)
        self._gamma = as_tensor_on(self._config.gamma, self._tau)
        self._one = as_tensor_on(1, self._tau)
        self._target_entropy = as_tensor_on(self._config.target_entropy, self._tau)

    @property
    def _alpha(self) -> torch.Tensor:
        """Get the alpha value with no gradient."""
        return self._log_alpha.exp().detach()

    def action(self, state: NDArray[ObsType], step: int) -> NDArray[ActTypeC]:
        """Sample an action from the actor network.

        Returns:
            action: The sampled action for training.
        """
        with torch.no_grad():
            a, log_prob = self._actor.sample(
                torch.from_numpy(state).float().to(self._config.device)
            )
            a = a.cpu()

        self._writer.log_action_stats(
            actions=a,
            data={"action/log_prob": log_prob},
            step=step,
            log_interval=self._config.log_interval,
            blocked=False,
        )
        return a.numpy()

    def update(self, experience: Experience, step: int) -> None:
        """Update the actor network.

        Steps:
        1. update critic. Formula:
            L_critic = mean((Qi(s, a) - target_q)^2, ...)
        2. update actor. Formula:
            L_actor = alpha * log_prob(a) - Q0(s, a)
        3. update alpha. Formula:
            L_alpha = -alpha * (log_prob(a) + target_entropy)
        4. soft update target networks. Formula:
            target_actor = tau * actor + (1 - tau) * target_actor
            target_critic = tau * critic + (1 - tau) * target_critic
        5. log the stats if necessary in background
        """
        exp = experience.to(self._config.device)

        # 1. update critic
        # 1.1 calculate target Q-value
        target_q = self._get_target_q(exp)
        # 1.2 get critic Q-value
        q = self._ctx.critic(exp.states, exp.actions)
        # Return Sequence: standard TD3/SAC, use Q1/Q2, usually train together
        # Return Single: the standard DDPG, use one Q.
        qs = q if isinstance(q, Sequence) else (q,)
        # 1.3 calculate critic loss
        assert qs[0].shape == target_q.shape, f"{qs[0].shape=}, {target_q.shape=}"
        critic_loss = torch.stack([nn.functional.mse_loss(qi, target_q) for qi in qs]).mean()
        # 1.4 update critic
        self._ctx.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self._config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self._ctx.critic.parameters(), self._config.max_grad_norm)
        self._ctx.critic_optimizer.step()

        # 2. update actor
        # 2.1 sample action in current policy
        new_action, log_prob = self._actor.sample(exp.states)
        # 2.2 get critic Q-value
        q_new = self._ctx.critic(exp.states, new_action)
        if isinstance(q_new, Sequence):
            q_new = q_new[0]
        # 2.3 calculate actor loss
        actor_loss = (self._alpha * log_prob - q_new).mean()
        # 2.4 update actor
        self._ctx.optimizer.zero_grad()
        actor_loss.backward()
        if self._config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
        self._ctx.optimizer.step()

        # 3. update alpha if necessary
        if self._config.auto_tune_alpha:
            # with torch.no_grad():
            #     _, log_prob_for_alpha, _ = self._actor.sample(exp.states)
            log_prob_for_alpha = log_prob.detach()
            alpha_loss = -(
                self._log_alpha.exp() * (log_prob_for_alpha + self._target_entropy).detach()
            ).mean()
            # update alpha
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        # 4. soft update target networks
        soft_update(self._ctx.critic, self._target_critic, self._tau)

        # log
        data = {
            "loss/actor": actor_loss,
            "loss/critic": critic_loss,
            "tune/log_prob": log_prob,
            "tune/alpha_value": self._alpha,
            "q_value/target": target_q,
            **{f"q_value/critic_{i}": qi for i, qi in enumerate(qs)},
        }
        if self._config.auto_tune_alpha:
            data["tune/alpha_loss"] = alpha_loss
            data["tune/target_entropy"] = self._target_entropy
        self._writer.log_stats(
            data=data,
            step=step,
            log_interval=self._config.log_interval,
            blocked=False,
        )

    def _get_target_q(self, experience: Experience) -> torch.Tensor:
        """Get the target Q-value for the given experience.

        Steps:
        1. get next action and log_prob from target actor
            Formula: a' = actor(s'), log_prob(a') = log_prob(actor(s'))
        2. get next Q-value from target critic
        3. calculate target Q-value with rewards, next Q-value, and entropy regularization.
            Formula:
                target_q = r + gamma * (1 - d) * (min(Qi(s', a'), ...) - alpha * log_prob(a'))

        Returns:
            target_q: The target Q-value. [batch, 1]
        """
        with torch.no_grad():
            # next action and log_prob from target actor
            next_a, next_log_prob = self._actor.sample(experience.next_states)

            # next Q-value from target critic
            target_critic_q = self._target_critic(experience.next_states, next_a)
            # Return Sequence: standard TD3/SAC, use Q1/Q2, usually train together
            # Return Single: the standard DDPG, use one Q.
            if isinstance(target_critic_q, Sequence):
                next_q: torch.Tensor = torch.min(*target_critic_q)
            else:
                next_q = target_critic_q

            # rewards shape [batch] -> [batch, 1]
            rewards = _to_batch2d(experience.rewards)
            dones = _to_batch2d(experience.dones)

            # calculate target Q-value with rewards, next Q-value, and entropy regularization
            target_q = rewards + self._gamma * (self._one - dones.float()) * (
                next_q - self._alpha * next_log_prob
            )
            if target_q.dim() == 1:
                target_q = target_q.unsqueeze(-1)
        return target_q


def _to_batch2d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [batch, 1].

    if [batch] unsqueeze to [batch, 1].
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    assert 1 == tensor.shape[1], f"{tensor.shape=}"
    return tensor
