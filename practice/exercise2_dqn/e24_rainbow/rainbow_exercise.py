import copy
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from gymnasium.spaces import Discrete
from numpy.typing import NDArray
from torch import Tensor

from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ObsType
from practice.exercise2_dqn.dqn_exercise import DQNConfig, DQNPod
from practice.exercise2_dqn.e24_rainbow.network import RainbowNet
from practice.exercise2_dqn.e24_rainbow.per_exercise import NStepReplay, PERBuffer, PERBufferConfig
from practice.utils_for_coding.image_utils import augment_image
from practice.utils_for_coding.writer_utils import CustomWriter


@dataclass(kw_only=True, frozen=True)
class RainbowConfig(DQNConfig):
    """Rainbow DQN Config."""

    noisy_std: float
    """The standard deviation of the noisy layer."""
    per_buffer_config: PERBufferConfig
    """The configuration for the prioritized experience replay buffer."""
    v_min: float
    """The minimum value of the value distribution."""
    v_max: float
    """The maximum value of the value distribution."""
    num_atoms: int
    """The number of atoms for the value distribution."""


class RainbowPod(DQNPod):
    """Rainbow DQN Pod.

    Rainbow DQN is a combination of:
    - Double DQN: https://arxiv.org/abs/1509.06461
    - Dueling DQN: https://arxiv.org/abs/1511.06581
    - Noisy DQN: https://arxiv.org/abs/1706.10295
    - Distributional DQN: https://arxiv.org/abs/1707.06887 (C51)
    - Prioritized Experience Replay Buffer: https://arxiv.org/abs/1511.05952 (PER)

    Reference:
    - https://arxiv.org/abs/1710.02298
    """

    def __init__(self, config: DQNConfig, ctx: ContextBase, writer: CustomWriter) -> None:
        # use DQNConfig for the base class
        assert isinstance(config, RainbowConfig)
        self._config: RainbowConfig = config
        self._ctx = ctx
        self._writer = writer
        self._noisy_std = config.noisy_std
        self._replay_buffer = PERBuffer(config.per_buffer_config)
        assert isinstance(self._ctx.network, RainbowNet)
        self._online_net = self._ctx.network
        self._target_net = copy.deepcopy(self._online_net)

        self._target_net.eval()
        self._online_net.train()

        # Get action space info
        assert isinstance(self._ctx.eval_env.action_space, Discrete)
        self._action_n = int(self._ctx.eval_env.action_space.n)
        self._delta_z = (config.v_max - config.v_min) / (config.num_atoms - 1)
        self._step = 0

    def sync_target_net(self) -> None:
        self._target_net.load_state_dict(self._online_net.state_dict())

    def action(self, states: NDArray[ObsType]) -> NDArray[ActType]:
        # Check if input is a single state or batch of states
        is_single = len(states.shape) == len(self._ctx.env_state_shape)
        state_batch = states if not is_single else states.reshape(1, *states.shape)
        states_tensor = torch.from_numpy(state_batch).to(self._config.device)

        self._online_net.reset_noise()
        with torch.no_grad():
            q_values = self._online_net(states_tensor).cpu()

        actions: NDArray[ActType] = q_values.argmax(dim=1).numpy().astype(ActType).reshape(-1)

        # Log stats
        self._writer.log_stats(
            data={
                "action/mean": actions.mean(),
                "action/std": actions.std(),
            },
            step=self._step,
            log_interval=self._config.log_interval,
        )

        self._step += 1
        return actions

    def update(self) -> None:
        """Update the network.

        Steps:
        1. sample batch from n-step PER buffer
        2. calculate the categorical projection
        3. compute the loss and update the network
        4. update the PER priority
        5. log stats
        """
        if len(self._replay_buffer) < self._config.batch_size:
            return

        # 1. sample batch from n-step PER buffer
        data, weights, data_idxs = self._replay_buffer.sample(self._config.batch_size)
        data = data.to(self._config.device)
        # augment the image
        states = augment_image(data.states) if (data.states.ndim == 4) else data.states

        # 2. calculate the m (categorical distribution)
        self._online_net.reset_noise()
        self._target_net.reset_noise()
        probs = self._online_net.forward_dist(states)
        # [B, action_n, atoms] -> [B, atoms]
        probs_a = _select_action_dist(probs, data.actions)
        # categorical distribution: [B, atoms]
        m = self._compute_m(data)

        # 3. compute the loss and update the network
        # [B, atoms] -> [B]
        loss_per_sample = -(m * probs_a.clamp(min=1e-6).log()).sum(dim=-1)
        w = torch.from_numpy(weights).to(self._config.device)
        weighted_loss = (loss_per_sample * w).mean()
        if not torch.isfinite(weighted_loss):
            raise ValueError("weighted_loss is not finite")
        self._ctx.optimizer.zero_grad()
        weighted_loss.backward()
        if self._config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._online_net.parameters(), self._config.max_grad_norm
            )
        self._ctx.optimizer.step()
        self._ctx.step_lr_schedulers()

        # 4. update the PER priority
        priorities = loss_per_sample.detach().abs().cpu().numpy()
        get_metrics = self._step % self._config.log_interval == 0
        per_metrics = self._replay_buffer.update_priorities(data_idxs, priorities, get_metrics)

        # 5. log stats
        self._writer.log_stats(
            data={
                "loss/original": loss_per_sample,
                "loss/weighted": weighted_loss,
                **per_metrics,
            },
            step=self._step,
            log_interval=self._config.log_interval,
            blocked=False,
        )

    def buffer_add(
        self,
        states: NDArray[ObsType],
        actions: NDArray[ActType],
        rewards: NDArray[np.float32],
        next_states: NDArray[ObsType],
        dones: NDArray[np.bool_],
        env_idxs: NDArray[np.int16],
    ) -> None:
        self._replay_buffer.add_batch(states, actions, rewards, next_states, dones, env_idxs)

    def _compute_m(self, data: NStepReplay) -> Tensor:
        """Compute the m (categorical distribution) for the given data.

        Returns:
            (B, Z) —— the m distribution for each sample
        """
        with torch.no_grad():
            next_states = (
                augment_image(data.n_next_states)
                if (data.n_next_states.ndim == 4)
                else data.n_next_states
            )
            q_next = self._online_net.forward(next_states)
            # [B, action_n] -> [B]
            a_next = q_next.argmax(dim=1)

            target_probs = self._target_net.forward_dist(next_states)
            # [B, action_n, atoms] -> [B, atoms]
            target_probs_a = _select_action_dist(target_probs, a_next)

            gamma_n = torch.pow(torch.full_like(data.rewards, self._config.gamma), data.n)
            m = _categorical_projection(
                next_prob=target_probs_a,
                rewards=data.rewards,
                dones=data.n_dones,
                gamma=gamma_n,
                # use cast for mypy
                support=cast(Tensor, self._online_net.support),
                v_min=self._config.v_min,
                v_max=self._config.v_max,
                delta_z=self._delta_z,
            )
        return m


def _select_action_dist(probs: Tensor, actions: Tensor) -> Tensor:
    """Select the action distribution from the C51 distribution.

    Args:
        probs:   (B, A, Z)  —— the C51 distribution p(z|s,a) after softmax
        actions: (B,)       —— the action index for each sample
    Returns:
        (B, Z) —— the action distribution for each sample
    """
    assert probs.dim() == 3, f"probs shape should be (B,A,Z), got {probs.shape}"
    b, _, z = probs.shape
    # [B] -> [B, 1, 1] -> [B, 1, Z]
    actions = actions.long().view(b, 1, 1).expand(-1, 1, z)
    # [B, A, Z] -> [B, 1, Z] -> [B, Z]
    return probs.gather(1, actions).squeeze(1)


def _categorical_projection(
    next_prob: Tensor,
    rewards: Tensor,
    dones: Tensor,
    gamma: Tensor | float,
    support: Tensor,
    v_min: float,
    v_max: float,
    delta_z: float,
) -> Tensor:
    """Categorical projection.

    Steps:
    1. calculate the Tz = r + (1-done) * gamma * z
    2. clip the Tz to [v_min, v_max]
    3. calculate the b, l, u
    4. initialize the target distribution m
    5. scatter the next_prob to m
    6. normalize the m

    Parameters
    ----------
    next_prob: Tensor
        The target/next distribution (after softmax), shape (B, atoms).
    rewards: Tensor
        The n-step accumulated rewards, shape (B,).
    dones: Tensor
        The termination flags, shape (B,).
    gamma: Tensor | float
        The discount factor, shape (B,) or scalar.
    support: Tensor
        The support of the value distribution, shape (atoms,).
    v_min: float
        The minimum value of the value distribution.
    v_max: float
        The maximum value of the value distribution.
    delta_z: float
        The width of the value distribution.

    Returns
    -------
    m: Tensor
        The target distribution, shape (B, atoms).

    Reference:
    - Bellemare et al., 2017 (C51)
    """
    # 0. data preparation
    device = next_prob.device
    dtype = next_prob.dtype
    B, atoms = next_prob.shape

    support = support.to(device=device, dtype=dtype)
    rewards = rewards.to(device=device, dtype=dtype).view(B, 1)
    dones = dones.to(device=device).view(B, 1).float()

    if isinstance(gamma, (float, int)):
        gamma = torch.full((B, 1), float(gamma), device=device, dtype=dtype)
    else:
        gamma = gamma.to(device=device, dtype=dtype).view(B, 1)

    # 1. calculate the Bellman target Tz = r + (1-done) * gamma * z: [B, atoms]
    Tz = rewards + (1.0 - dones.float()) * gamma * support.view(1, atoms)

    # 2. clip the Tz to [v_min, v_max]
    Tz = Tz.clamp(v_min, v_max)

    # 3. calculate the b, l, u
    b = (Tz - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    # 4. initialize the target distribution m
    m = torch.zeros(B, atoms, device=device, dtype=dtype)

    # 5. scatter the next_prob to l and u
    #    weight: (u - b) for l, (b - l) for u
    w_l = next_prob * (u.type_as(b) - b)
    w_u = next_prob * (b - l.type_as(b))

    # 6. normalize the m
    # 6.1 clip the index to the valid range
    l = l.clamp(0, atoms - 1)
    u = u.clamp(0, atoms - 1)
    # 6.2 aggregate the next_prob to m
    m.scatter_add_(dim=1, index=l, src=w_l)
    m.scatter_add_(dim=1, index=u, src=w_u)
    # 6.3 normalize the m with robust handling of zero sums
    m_sum = m.sum(dim=1, keepdim=True)
    # Handle cases where the sum becomes zero (numerical issues)
    zero_sum_mask = m_sum <= 1e-8
    if torch.any(zero_sum_mask):
        # For zero sum cases, create uniform distribution as fallback
        uniform_prob = 1.0 / atoms
        m[zero_sum_mask.squeeze(-1)] = uniform_prob
        m_sum = m.sum(dim=1, keepdim=True)

    # Normalize with additional safety margin
    m = m / m_sum.clamp_min(1e-8)

    return m
