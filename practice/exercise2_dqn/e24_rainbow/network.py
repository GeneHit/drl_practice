import math
from typing import cast

import torch
import torch.nn as nn

from practice.base.env_typing import ActType
from practice.utils_for_coding.network_utils import MLP
from practice.utils_for_coding.numpy_tensor_utils import argmax_action


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


class QNetCNN(nn.Module):
    """Q network with 2D convolution.

    Same as DeepMind's DQN paper for Atari:
    https://www.nature.com/articles/nature14236
    """

    def __init__(self, in_shape: tuple[int, int, int], fc_size: int = 256) -> None:
        super().__init__()
        c, h, w = in_shape
        # convolution layer sequence
        self.conv = nn.Sequential(
            # paper: 8, s4; use 7, s2
            nn.Conv2d(c, 32, kernel_size=7, stride=2),
            nn.ReLU(),
            # paper: 4, s2; use 5, s2
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # calculate the output size of the convolution layer
        with torch.no_grad():
            # create a mock input (batch=1, c, h, w)
            test_input = torch.zeros(1, c, h, w)
            conv_output = self.conv(test_input)
            conv_output_size = conv_output.size(1)

        # full connected layer
        self.fc = nn.Sequential(
            nn.LayerNorm(conv_output_size),
            nn.Linear(conv_output_size, fc_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use cast to make mypy happy
        return cast(torch.Tensor, self.fc(self.conv(x / 255.0)))


class RainbowNet(nn.Module):
    """Rainbow Network.

    Rainbow network is a combination of:
    - dueling: https://arxiv.org/abs/1511.06581
    - noisy: https://arxiv.org/abs/1706.10295
    - distributional (C51): https://arxiv.org/abs/1707.06887
    """

    def __init__(
        self,
        state_n: int | tuple[int, int, int],
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
        if isinstance(state_n, tuple):
            assert len(hidden_sizes) == 1, "hidden_sizes should be [fc_size] for CNN"
            last_size = hidden_sizes[0]
            self.feature: nn.Module = QNetCNN(state_n, last_size)
        else:
            last_size = hidden_sizes[-1]
            self.feature = MLP(
                input_dim=state_n, hidden_sizes=hidden_sizes[:-1], output_dim=last_size
            )

        # C51 distributional head
        support = torch.linspace(v_min, v_max, num_atoms)  # (num_atoms,)
        self.register_buffer("support", support)

        # value / advantage heads (Dueling + Noisy + C51)
        self.value_head = NoisyLinear(last_size, num_atoms, std=noisy_std)
        self.advantage_head = NoisyLinear(last_size, action_n * num_atoms, std=noisy_std)

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
