import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader

from practice.utils_for_coding.network_utils import MLP


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Training config for dynamics models."""

    epoches: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    loss_weight_delta: float = 1.0  # weight for Δs NLL
    loss_weight_reward: float = 1.0  # weight for r NLL
    loss_weight_done: float = 1.0  # weight for BCE(done)
    val_ratio: float = 0.1
    early_stop_patience: int = 10
    bootstrap: bool = True  # bootstrap per model (with replacement)


@dataclass(frozen=True, kw_only=True)
class ModelBasedConfig:
    """Config consumed by ModelBasedEnv (includes TrainConfig)."""

    num_models: int
    train: TrainConfig
    done_threshold: float = 0.5
    log_std_bounds: tuple[float, float] = (-5.0, 2.0)
    eps: float = 1e-6


class EnvModel(nn.Module):
    """Predict the [Δs,r], and done's logit."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # MLP signature: input_dim, output_dim, hidden_sizes
        self.backbone = MLP(
            input_dim=state_dim + action_dim,
            output_dim=hidden_sizes[-1],
            hidden_sizes=hidden_sizes[:-1],
        )
        self.head_gauss = nn.Linear(hidden_sizes[-1], 2 * (state_dim + 1))
        self.head_done = nn.Linear(hidden_sizes[-1], 1)
        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                nn.init.zeros_(m.bias)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the [Δs,r], and done's logit.

        Args:
            state: (B, state_dim)
            action: (B, action_dim)

        Returns:
            mean: (B, state_dim+1) for [Δs, r]
            log_std: (B, state_dim+1)
            done_logit: (B, 1)
        """
        x = torch.cat([state, action], dim=-1)
        h = self.backbone(x)
        gauss = self.head_gauss(h)
        mean, log_std = torch.chunk(gauss, 2, dim=-1)
        done_logit = self.head_done(h)
        return mean, log_std, done_logit


class ModelBasedEnv:
    """A model-based environment wrapper that holds an ensemble (list) of dynamics models."""

    def __init__(self, model: EnvModel, cfg: ModelBasedConfig) -> None:
        self._models = [model, *[copy.deepcopy(model) for _ in range(cfg.num_models - 2)]]
        self._optimizers = [
            torch.optim.AdamW(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
            )
            for model in self._models
        ]
        self._cfg = cfg

        # Infer state/action dims from the first model (all models should match).
        self._state_dim = int(model.state_dim)
        self._action_dim = int(model.action_dim)

        # Normalization buffers (set via set_normalizer or set_rollout_context).
        self._mu_in: Optional[torch.Tensor] = None  # shape (state_dim + action_dim,)
        self._std_in: Optional[torch.Tensor] = None
        self._mu_out: Optional[torch.Tensor] = None  # shape (state_dim + 1,) for [Δs, r]
        self._std_out: Optional[torch.Tensor] = None

        # Rollout selector
        self._rollout_model_index: int = 0

    def _set_normalizer(
        self,
        mu_in: torch.Tensor,
        std_in: torch.Tensor,
        mu_out: torch.Tensor,
        std_out: torch.Tensor,
    ) -> None:
        """Set z-score stats for inputs [s,a] and outputs [Δs,r]."""
        device = next(self._models[0].parameters()).device
        self._mu_in = mu_in.to(device)
        self._std_in = std_in.clamp_min(self._cfg.eps).to(device)
        self._mu_out = mu_out.to(device)
        self._std_out = std_out.clamp_min(self._cfg.eps).to(device)

    def set_rollout_model(self) -> None:
        """Choose which model(s) to use for rollout."""
        self._rollout_model_index = random.randint(0, self._cfg.num_models - 1)

    @torch.no_grad()
    def step(
        self, state: torch.Tensor, action: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict (next_state, reward, done) using the selected rollout model(s).

        Args:
            state:  (B, state_dim)
            action: (B, action_dim)

        Returns:
            next_state: (B, state_dim)
            reward:     (B, 1)
            done:       (B, 1) in {0., 1.}
        """
        assert state.ndim == 2 and action.ndim == 2, "state/action must be (B, D)"

        # Type narrowing for normalizer tensors
        assert (
            self._mu_in is not None
            and self._std_in is not None
            and self._mu_out is not None
            and self._std_out is not None
        ), "Normalizer not set, should call train() first"

        x = torch.cat([state, action], dim=-1)
        x_norm = (x - self._mu_in) / self._std_in
        s_norm, a_norm = x_norm[:, : self._state_dim], x_norm[:, self._state_dim :]

        mean, log_std, done_logit = self._models[self._rollout_model_index](s_norm, a_norm)

        # Stabilize log_std and sample/mean in normalized space
        log_std = torch.clamp(log_std, self._cfg.log_std_bounds[0], self._cfg.log_std_bounds[1])
        if not deterministic:
            y_norm = Normal(mean, log_std.exp()).rsample()
        else:
            y_norm = mean

        # Denormalize [Δs, r]
        y = y_norm * self._std_out + self._mu_out
        delta_s = y[:, : self._state_dim]
        reward = y[:, self._state_dim : self._state_dim + 1]

        next_state = state + delta_s
        done = (torch.sigmoid(done_logit) > self._cfg.done_threshold).to(torch.bool)
        return next_state, reward, done

    def train(self, dataloader: DataLoader[dict[str, torch.Tensor]]) -> dict[str, list[float]]:
        """Train ALL models using Gaussian NLL for [Δs,r] and BCE for done.

        - Computes z-score stats from the real dataset and sets them.
        - Supports validation split and early stopping.
        - If an optimizer is provided, it should include ALL models' params; otherwise, one is created.

        Returns a dict of training/validation losses per epoch (averaged across models).
        """
        raise NotImplementedError("Not implemented")
