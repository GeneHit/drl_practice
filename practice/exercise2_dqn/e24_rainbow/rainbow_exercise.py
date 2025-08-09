import copy
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Discrete
from numpy.typing import NDArray
from torch import Tensor

from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ObsType
from practice.exercise2_dqn.dqn_exercise import DQNConfig, DQNPod
from practice.exercise2_dqn.e24_rainbow.per_exercise import NStepReplay, PERBuffer, PERBufferConfig
from practice.utils_for_coding.network_utils import MLP
from practice.utils_for_coding.numpy_tensor_utils import argmax_action
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


class NoisyLinear(nn.Module):
    """Noisy Linear Layer with factorized Gaussian noise.

    Equation:
    w = μ_w + sigma_w @ (f(ε_out) @ f(ε_in)),  b = μ_b + sigma_b @ f(ε_out)
    where f(x) = sign(x) * sqrt(|x|)
    It means:
    - w = μ_w + sigma_w @ (f(ε_out) @ f(ε_in))
    - b = μ_b + sigma_b @ f(ε_out)
    - f(x) = sign(x) * sqrt(|x|)

    Reference:
    - https://arxiv.org/abs/1706.10295
    - https://arxiv.org/abs/1707.06887
    """

    def __init__(self, in_features: int, out_features: int, std: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # learnable parameters: μ、σ
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # register_buffer for current noise samples (no gradient)
        self.register_buffer("eps_in", torch.zeros(in_features))
        self.register_buffer("eps_out", torch.zeros(out_features))

        self.reset_parameters(std)
        self.reset_noise()

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        # f(ε) = sign(ε) * sqrt(|ε|)
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_parameters(self, std: float) -> None:
        # μ ~ U[-1/sqrt(in), 1/sqrt(in)]
        mu_bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_bound, mu_bound)
        nn.init.uniform_(self.bias_mu, -mu_bound, mu_bound)
        # σ = std / sqrt(in_features)
        sigma_init = std / math.sqrt(self.in_features)
        nn.init.constant_(self.weight_sigma, sigma_init)
        nn.init.constant_(self.bias_sigma, sigma_init)

    def reset_noise(self) -> None:
        # sample factorized noise
        self.eps_in = self._scale_noise(self.in_features).to(self.weight_mu.device)
        self.eps_out = self._scale_noise(self.out_features).to(self.weight_mu.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # training: use noisy weights and biases
            w_noise = torch.outer(self.eps_out, self.eps_in)  # (out, in)
            w = self.weight_mu + self.weight_sigma * w_noise
            b = self.bias_mu + self.bias_sigma * self.eps_out
        else:
            # evaluation: use mean parameters (no noise)
            w = self.weight_mu
            b = self.bias_mu
        return nn.functional.linear(x, w, b)


class RainbowNet(nn.Module):
    """Rainbow Network.

    Rainbow network is a combination of:
    - dueling: https://arxiv.org/abs/1511.06581
    - noisy: https://arxiv.org/abs/1706.10295
    - distributional (C51): https://arxiv.org/abs/1707.06887
    """

    def __init__(
        self,
        state_n: int,
        action_n: int,
        hidden_sizes: tuple[int, ...],
        *,
        noisy_std: float = 0.5,
        v_min: float = -10.0,
        v_max: float = 10.0,
        num_atoms: int = 51,
    ) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.action_n = action_n

        # feature stream
        self.feature = MLP(
            input_dim=state_n, hidden_sizes=hidden_sizes[:-1], output_dim=hidden_sizes[-1]
        )

        # C51 distributional head
        support = torch.linspace(v_min, v_max, num_atoms)  # (num_atoms,)
        self.register_buffer("support", support)

        # value / advantage heads (Dueling + Noisy + C51)
        self.value_head = NoisyLinear(hidden_sizes[-1], num_atoms, std=noisy_std)
        self.advantage_head = NoisyLinear(hidden_sizes[-1], action_n * num_atoms, std=noisy_std)

    def reset_noise(self) -> None:
        """Reset the noise before forward pass when training.

        Usually, during training, acting and learning call this function once.
        """
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the q value.

        Returns:
            q: The expected Q(s), shape (B, action_n).
        """
        # 1. get the C51 distribution: [B, action_N, atoms_Z]
        probs = self.forward_dist(x)
        # 2. expected Q = sum_z p(z|s,a)*z: [B, action_n]
        # q[b,n] = sum_z probs[b,n,z] * support[z]
        q = torch.einsum("bnz,z->bn", probs, self.support)
        return q

    def action(self, x: torch.Tensor) -> ActType:
        """Get the action for evaluation/gameplay with 1 environment.

        Returns:
            action: The single action.
        """
        # greedy strategy
        return argmax_action(self.forward(x), dtype=ActType)

    def forward_dist(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the C51 distribution.

        Returns:
            probs: The C51 distribution p(z|s,a), shape (B, action_n, num_atoms).
        """
        h = self.feature(x)  # (B, hidden)

        # value branch: (B, atoms)
        v_logits = self.value_head(h)
        # advantage branch: (B, action_n * atoms) -> (B, action_n, atoms)
        a_logits = self.advantage_head(h).view(-1, self.action_n, self.num_atoms)

        # dueling combination (combine in logits level first, then softmax for stability)
        # Q_logits(s,a,·) = V_logits(s,·) + A_logits(s,a,·) - mean_a A_logits(s,a,·)
        a_mean = a_logits.mean(dim=1, keepdim=True)  # (B, 1, atoms)
        q_logits = v_logits.unsqueeze(1) + (a_logits - a_mean)  # (B, action_n, atoms)

        # softmax for each (s,a) atom dimension, get distribution
        probs = nn.functional.softmax(q_logits, dim=-1).clamp(min=1e-6)  # stability
        return probs


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

        # 2. calculate the m (categorical distribution)
        self._online_net.reset_noise()
        self._target_net.reset_noise()
        probs = self._online_net.forward_dist(data.states)
        # [B, action_n, atoms] -> [B, atoms]
        probs_a = _select_action_dist(probs, data.actions)
        # categorical distribution: [B, atoms]
        m = self._compute_m(data)

        # 3. compute the loss and update the network
        # [B, atoms] -> [B]
        loss_per_sample = -(m * probs_a.clamp(min=1e-6).log()).sum(dim=-1)
        w = torch.from_numpy(weights).to(self._config.device)
        weighted_loss = (loss_per_sample * w).mean()
        self._ctx.optimizer.zero_grad()
        weighted_loss.backward()
        self._ctx.optimizer.step()

        # 4. update the PER priority
        priorities = loss_per_sample.detach().abs().cpu().numpy()
        self._replay_buffer.update_priorities(data_idxs, priorities)

        # 5. log stats
        self._writer.log_stats(
            data={
                "loss/original": loss_per_sample,
                "loss/weighted": weighted_loss,
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
            q_next = self._online_net.forward(data.n_next_states)
            # [B, action_n] -> [B]
            a_next = q_next.argmax(dim=1)

            target_probs = self._target_net.forward_dist(data.n_next_states)
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
    # 6.3 normalize the m
    m = m / m.sum(dim=1, keepdim=True).clamp_min(1e-6)

    return m
