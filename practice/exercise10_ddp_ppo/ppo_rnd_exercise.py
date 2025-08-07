from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import ActTypeC, ObsType
from practice.base.rewarder import RewardBase, RewardConfig
from practice.base.trainer import TrainerBase
from practice.exercise7_ppo.ppo_exercise import _RolloutBuffer, _StepData
from practice.utils.dist_utils import unwrap_model
from practice.utils_for_coding.network_utils import MLP, LogStdHead, init_weights
from practice.utils_for_coding.numpy_tensor_utils import np2tensor, tensor2np_1d
from practice.utils_for_coding.rollout_utils import get_good_transition_mask
from practice.utils_for_coding.scheduler_utils import ScheduleBase
from practice.utils_for_coding.writer_utils import CustomWriter


class ContACNet(nn.Module):
    """Continuous Actor-Critic Network.

    Features:
    - for continuous action space.
    - layers:
        - shared feature layer
        - policy head: output the mean and logstd of the action.
        - value head: output the value of the state.
    Key methods:
    - forward: output the action when evaluating.
    - sample: sample an action when training.

    Args:
        obs_dim: The dimension of the observation.
        act_dim: The dimension of the action.
        hidden_sizes: The hidden sizes of the MLP.
        action_scale: The scale of the action.
        action_bias: The bias of the action.
        log_std_min: The minimum logstd of the action.
        log_std_max: The maximum logstd of the action.
        use_layer_norm: Whether to use layer normalization.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        action_scale: float,
        action_bias: float,
        use_layer_norm: bool = False,
        log_std_min: float = -20,
        log_std_max: float = 2,
        log_std_state_dependent: bool = False,
    ) -> None:
        super().__init__()
        # shared feature layer
        self.shared_net = MLP(
            input_dim=obs_dim,
            output_dim=hidden_sizes[-1],
            hidden_sizes=hidden_sizes[:-1],
            use_layer_norm=use_layer_norm,
        )
        # policy head
        self.policy_mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.policy_logstd = LogStdHead(
            input_dim=hidden_sizes[-1],
            output_dim=act_dim,
            state_dependent=log_std_state_dependent,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        # value head
        self.value = nn.Linear(hidden_sizes[-1], 1)
        init_weights(self.policy_mean)
        init_weights(self.value)

        # for type checking
        self.action_scale: Tensor
        self.action_bias: Tensor
        # register the parameters
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))

    def action(self, obs: Tensor) -> NDArray[ActTypeC]:
        """Get the action for evaluation/gameplay with 1 environment.

        Returns:
            action: The deterministic action for one environment.
        """
        with torch.no_grad():
            x = self.shared_net(obs)
            mean = self.policy_mean(x)
            action: Tensor = torch.tanh(mean) * self.action_scale + self.action_bias
            return tensor2np_1d(action, dtype=ActTypeC())

    def forward(
        self, obs: Tensor, actions: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample an action when training.

        Args:
            obs: The observation.
            actions: The actions to be sampled.
                If not None, the actions will be used to compute the log_prob, and return the same
                action.

        Returns:
            action: The sampled continuous action.
            log_prob: The log probability of the action.
            value: The value of the state.
            entropy: The entropy of the action.
        """
        # 1. shared feature layer
        x = self.shared_net(obs)

        # 2. get the normal distribution
        # 2.1 get policy mean and log_std
        mean = self.policy_mean(x)
        log_std = self.policy_logstd(x, mean)
        # 2.2 get normal distribution with mean and std
        normal = torch.distributions.Normal(mean, log_std.exp())

        # 3. value and entropy
        value = self.value(x)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)

        # 4. calculate the log_prob with provided actions
        eps = 1e-6
        if actions is not None:
            action_unscaled = (actions - self.action_bias) / self.action_scale
            action_unscaled = torch.clamp(action_unscaled, -1 + eps, 1 - eps)
            pre_tanh = 0.5 * (torch.log1p(action_unscaled) - torch.log1p(-action_unscaled))
            log_prob: Tensor = normal.log_prob(pre_tanh) - torch.log(
                self.action_scale * (1 - action_unscaled.pow(2)) + eps
            )
            log_prob = log_prob.sum(dim=-1, keepdim=True)  # [batch, 1]
            return actions, log_prob, value, entropy

        # 5. sample the action
        # 5.1 reparameterization trick (mean + std * N(0,1))
        z = normal.rsample()
        # 5.2 scale to environment action space
        action = torch.tanh(z)
        action_scaled = action * self.action_scale + self.action_bias
        # 5.3 calculate log_prob
        log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1 - action.pow(2)) + eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # [batch, 1]

        return action_scaled, log_prob, value, entropy


@dataclass(frozen=True, kw_only=True)
class ContPPOConfig(BaseConfig):
    """The configuration for the continuous PPO algorithm."""

    timesteps: int
    """The step number of the total rollouts (or main train loop, or update number).

    The total data = timesteps * rollout_len * vector_env_num.
    The rollout is a sequence of states, actions, rewards, values, log_probs, dones.
    """

    rollout_len: int
    """The length of the rollout."""

    gamma: float
    """The discount factor."""

    gae_lambda: float
    """The lambda for the GAE."""

    entropy_coef: ScheduleBase
    """The entropy coefficient for the entropy loss."""

    value_loss_coef: float = 0.5
    """The coefficient for the value loss."""

    max_grad_norm: float | None = None
    """The maximum gradient norm for gradient clipping."""

    critic_lr: float
    """The learning rate for the critic."""

    num_epochs: int
    """The number of epochs to update the policy."""

    minibatch_num: int
    """The number of the minibatch."""

    clip_coef: float | ScheduleBase
    """The clip coefficient for the PPO."""

    value_clip_range: float | None = None
    """The clip range for the value function."""

    reward_clip_range: ScheduleBase | None = None
    """The clip range for the original env reward."""

    reward_configs: tuple[RewardConfig, ...] = ()
    """The reward configurations."""

    action_scale: float
    """The scale of the action."""

    action_bias: float
    """The bias of the action."""

    log_std_min: float
    """The minimum logstd of the action."""

    log_std_max: float
    """The maximum logstd of the action."""

    use_layer_norm: bool = False
    """Whether to use layer normalization."""

    hidden_sizes: Sequence[int]
    """The hidden sizes of the Actor-Critic MLP."""

    log_std_state_dependent: bool
    """Whether to use state-dependent logstd.

    If True, the logstd is a linear function of the state.
    If False, the logstd is a constant.
    """


class ContPPOTrainer(TrainerBase):
    """The trainer for the continuous PPO algorithm."""

    def __init__(self, config: ContPPOConfig, ctx: ContextBase) -> None:
        super().__init__(config, ctx)
        self._config: ContPPOConfig = config  # make mypy happy
        self._rewarders = tuple(rewarder.get_rewarder() for rewarder in self._config.reward_configs)

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
        envs = self._ctx.continuous_envs
        pod = _ContPPOPod(config=self._config, ctx=self._ctx, writer=writer)

        # Create variables for loop
        episode_num = 0
        global_step = 0
        prev_dones: NDArray[np.bool_] = np.zeros(envs.num_envs, dtype=np.bool_)
        assert self._config.env_config.vector_env_num is not None
        state, _ = envs.reset()
        for _ in tqdm(range(self._config.timesteps), desc="Rollouts"):
            # 2. Run one rollout
            for _ in range(self._config.rollout_len):
                # sample action and buffer partial data
                actions = pod.action(state)

                # step environment
                # we use AutoReset wrapper, so the envs will be reset automatically when it's done
                # when any done in n step, the next_states of n+1 step is the first of the next episode
                next_states, rewards, terms, truncs, infos = envs.step(actions)
                dones = np.logical_or(terms, truncs)

                # add extra reward
                total_rewards = _add_extra_reward(
                    rewarders=self._rewarders,
                    env_rewards=rewards.astype(np.float32),
                    states=state,
                    next_states=next_states,
                    global_step=global_step,
                    writer=writer,
                    log_interval=100,
                    prev_dones=prev_dones,
                )

                # buffer the data after stepping the environment
                # when any pre done, it will buffer a bad transition between two episodes.
                pod.add_stepped_data(next_states=next_states, rewards=total_rewards, dones=dones)

                # update state
                state = next_states
                global_step += 1
                prev_dones = dones
                # record the episode data
                episode_num += writer.log_episode_stats_if_has(infos, episode_num)

            # 3. Update the policy/value network
            pod.update()
            # 4. Clear the buffer
            pod.reset()

        writer.close()


class _ContPPOPod:
    """The PPO pod for the continuous PPO algorithm.

    The actor_critic network should outputs: action, log_prob and value.
    """

    def __init__(self, config: ContPPOConfig, ctx: ContextBase, writer: CustomWriter) -> None:
        self._config = config
        self._ctx = ctx
        self._writer = writer
        self._rollout: _RolloutBuffer = _RolloutBuffer()
        self._actor_critic = ctx.network
        assert isinstance(unwrap_model(self._actor_critic), ContACNet), (
            "The network must be a ContACNet"
        )
        self._rewarders = [rewarder.get_rewarder() for rewarder in self._config.reward_configs]
        self._rollout_count: int = 0
        self._pre_last_dones: Tensor | None = None

        self._actor_critic.train()

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

    def action(self, states: NDArray[ObsType]) -> NDArray[ActTypeC]:
        """Get the actions of the policy and buffer partial data.

        Args:
            states: The states to get the action of.

        Returns:
            actions: The continuous actions.
        """
        # get policy logits and value
        states_tensor = torch.from_numpy(states).float().to(self._config.device)
        with torch.no_grad():
            actions, log_probs, values, _ = self._actor_critic(states_tensor)

        # convert to numpy array for envs
        actions_np: NDArray[ActTypeC] = actions.cpu().numpy().astype(ActTypeC)

        # buffer the partial data
        self._rollout.add_before_acting(states, actions_np, log_probs, values)
        return actions_np

    def update(self) -> None:
        """Update the actor and critic."""
        # 1. Get the data from the buffer
        rollout = self._rollout.get_data()

        # 2. Compute the advantages [d, ] (d: the valid data length)
        advantages, returns = self._compute_advantages(rollout)
        advantages, log_probs, returns, states, actions, values = self._filter_bad_transition(
            rollout=rollout, advantages=advantages, returns=returns
        )
        # normalize the advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Update the policy and value network
        config = self._config
        batch_size = advantages.shape[0]

        assert config.num_epochs > 0, "The number of epochs must be greater than 0."
        for _ in range(config.num_epochs):
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            minibatches = np.array_split(indices, config.minibatch_num)
            for mb_inds in minibatches:
                # 3.1 Update the policy network
                _, log_probs_new, values_pred, entropy = self._actor_critic(
                    states[mb_inds], actions[mb_inds]
                )

                # [n, 1] -> [n, ]
                ratio = torch.exp((log_probs_new - log_probs[mb_inds]).view(-1))
                unclipped_pg_loss = ratio * advantages[mb_inds]
                clip_coef = (
                    config.clip_coef
                    if isinstance(config.clip_coef, float)
                    else config.clip_coef(self._rollout_count)
                )
                clipped_pg_loss = (
                    torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages[mb_inds]
                )
                pg_loss = -torch.mean(torch.min(unclipped_pg_loss, clipped_pg_loss))

                # 3.2 Update the value network
                value_mse = self._value_mse_loss(
                    value_old=values[mb_inds], values_pred=values_pred, returns=returns[mb_inds]
                )
                value_loss = config.value_loss_coef * value_mse

                entropy_coef = config.entropy_coef(self._rollout_count)
                entropy = entropy.mean()
                entropy_loss = -entropy_coef * entropy

                # 3.3 Update the total loss
                total_loss = pg_loss + value_loss + entropy_loss

                # 3.4 Backward and update the parameters
                self._ctx.optimizer.zero_grad()
                total_loss.backward()
                if config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), config.max_grad_norm
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

    def _compute_advantages(self, rollout: Sequence[_StepData]) -> tuple[Tensor, Tensor]:
        """Compute the advantages.

        Parameters
        ----------
            rollout: The rollout data.

        Returns
        -------
            advantages: The advantages.
            returns: The returns.
        """
        t_1_data = rollout[-1]
        # total steps T, where rollout timestamp is 0, 1, ..., T-1
        t = len(rollout)
        device = self._config.device
        gamma = self._config.gamma
        gae_lambda = self._config.gae_lambda

        # get the value of the last step V_T:
        # 1) get the V(s_T) of all envs: [batch, 1]
        states_tensor = np2tensor(t_1_data.states).to(device)
        with torch.no_grad():
            _, _, v_all, _ = self._actor_critic(states_tensor)
        # 2) mask the done envs: done→0, not done→V(s_T)
        done = torch.from_numpy(t_1_data.dones).to(device)
        v_t = v_all.view(-1) * (~done)

        # stack the data to [T, N]
        rewards = torch.stack([np2tensor(step.rewards) for step in rollout]).to(device)
        if self._config.reward_clip_range is not None:
            reward_clip_range = self._config.reward_clip_range(self._rollout_count)
            rewards = torch.clamp(rewards, -reward_clip_range, reward_clip_range)
        dones = torch.stack([np2tensor(step.dones) for step in rollout]).to(device)
        # value has been on device
        values = torch.stack([step.values.view(-1) for step in rollout])

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

        return advantages, returns

    def _filter_bad_transition(
        self, rollout: Sequence[_StepData], advantages: Tensor, returns: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Filter the bad transition.

        See _RolloutBuffer for the reason and more details.

        Have to throw away the data of the bad transition between two episodes, don't use its data
        and compute graph of backward.

        Parameters
        ----------
            rollout: The rollout data.
            advantages: The advantages.
            returns: The returns.

        Returns
        -------
        The valid data of the rollout (d (<= T * N) is the valid data length):
            advantages: Tensor, shape [d, ]
            log_probs: Tensor, shape [d, ]
            returns: Tensor, shape [d, ]
            states: Tensor, shape [d, obs_dim]
            actions: Tensor, shape [d, ]
            values: Tensor, shape [d, ]
        """
        device = self._config.device

        # stack the data. TODO: define a function to stack the data.
        # log_probs has been on device
        log_probs = torch.stack([step.log_probs for step in rollout])
        states = torch.stack([np2tensor(step.states) for step in rollout]).to(device)
        actions = torch.stack([np2tensor(step.actions) for step in rollout]).to(device)
        dones = torch.stack([torch.from_numpy(step.dones) for step in rollout]).to(device)
        values = torch.stack([step.values for step in rollout]).to(device)

        # get the good transition mask: pre_dones != 1
        if self._pre_last_dones is None:
            self._pre_last_dones = torch.zeros_like(dones[-1], dtype=torch.bool)
        good_mask = get_good_transition_mask(dones, self._pre_last_dones)
        self._pre_last_dones = dones[-1]

        # filter the bad transition between two episodes. dim: [T, N] -> [d, ]
        assert good_mask.ndim == 2
        mask_flat = good_mask.view(-1)
        advantages = advantages.flatten(0, 1)[mask_flat]
        log_probs = log_probs.flatten(0, 1)[mask_flat]
        returns = returns.flatten(0, 1)[mask_flat]
        # dims of actions is [T, N, act_dim] -> [d, act_dim]
        actions = actions.flatten(0, 1)[mask_flat]
        # dims of states is [T, N, obs_dim] -> [d, obs_dim]
        states = states.flatten(0, 1)[mask_flat]
        values = values.flatten(0, 1)[mask_flat]

        return advantages, log_probs, returns, states, actions, values

    def _value_mse_loss(self, value_old: Tensor, values_pred: Tensor, returns: Tensor) -> Tensor:
        """Compute the value MSE loss.

        Clip the value if configured.

        Args:
            value_old: The old value.
            values_pred: The predicted value.
            returns: The returns.
        """
        if self._config.value_clip_range is None:
            return nn.functional.mse_loss(values_pred.view(-1), returns)

        value_mse = nn.functional.mse_loss(values_pred.view(-1), returns, reduction="none")
        values_pred_clipped = value_old + torch.clamp(
            values_pred - value_old, -self._config.value_clip_range, self._config.value_clip_range
        )
        value_mse_clipped = nn.functional.mse_loss(
            values_pred_clipped.view(-1), returns, reduction="none"
        )
        return 0.5 * torch.max(value_mse, value_mse_clipped).mean()


def _add_extra_reward(
    rewarders: Sequence[RewardBase],
    env_rewards: NDArray[np.float32],
    states: NDArray[ObsType],
    next_states: NDArray[ObsType],
    global_step: int,
    writer: CustomWriter,
    log_interval: int,
    prev_dones: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Compute the reward with the extra rewarders.

    Args:
        rewarders: The rewarders.
        env_rewards: The environment rewards, shape [v,].
        states: The states, shape [v, obs_dim].
        next_states: The next states, shape [v, obs_dim].
        global_step: The global step.
        writer: The writer.
        log_interval: The log interval.

    Returns:
        rewards: The reward = env reward + extra rewarders' reward, shape [v,]
    """
    pre_non_terminal_mask = ~prev_dones
    if not np.any(pre_non_terminal_mask):
        return env_rewards

    states = states[pre_non_terminal_mask]
    next_states = next_states[pre_non_terminal_mask]
    rewards = env_rewards[pre_non_terminal_mask]

    extra_rewards: list[NDArray[np.float32]] = [
        rewarder.get_reward(states, next_states, global_step) for rewarder in rewarders
    ]
    env_reward_mean = env_rewards.max()
    if extra_rewards:
        rewards += sum(extra_rewards)

        writer.log_stats(
            data={
                "reward/env": env_reward_mean,
                **{
                    f"reward/{rewarder.__class__.__name__}": extra_rewards[idx].max()
                    for idx, rewarder in enumerate(rewarders)
                },
            },
            step=global_step,
            log_interval=log_interval,
        )

    env_rewards[pre_non_terminal_mask] = rewards
    return env_rewards
