import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader

from practice.utils_for_coding.buffer import Experience
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
    rollout_mode: Literal["random", "mean"] = "random"


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
        self.models = [model, *[copy.deepcopy(model) for _ in range(cfg.num_models - 1)]]
        self._optimizers = [
            torch.optim.AdamW(
                model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
            )
            for model in self.models
        ]
        self.cfg = cfg

        # Infer state/action dims from the first model (all models should match).
        m0 = self.models[0]
        assert hasattr(m0, "state_dim") and hasattr(m0, "action_dim"), (
            "Each model must expose state_dim and action_dim."
        )
        self.state_dim = int(m0.state_dim)
        self.action_dim = int(m0.action_dim)

        # Normalization buffers (set via set_normalizer or set_rollout_context).
        self.mu_in: Optional[torch.Tensor] = None  # shape (state_dim + action_dim,)
        self.std_in: Optional[torch.Tensor] = None
        self.mu_out: Optional[torch.Tensor] = None  # shape (state_dim + 1,) for [Δs, r]
        self.std_out: Optional[torch.Tensor] = None

        # Rollout selector
        self.rollout_mode: Literal["random", "mean"] = cfg.rollout_mode
        self.rollout_index: int = 0

    @torch.no_grad()
    def set_normalizer(
        self,
        mu_in: torch.Tensor,
        std_in: torch.Tensor,
        mu_out: torch.Tensor,
        std_out: torch.Tensor,
    ) -> None:
        """Set z-score stats for inputs [s,a] and outputs [Δs,r]."""
        device = next(self.models[0].parameters()).device
        self.mu_in = mu_in.to(device)
        self.std_in = std_in.clamp_min(self.cfg.eps).to(device)
        self.mu_out = mu_out.to(device)
        self.std_out = std_out.clamp_min(self.cfg.eps).to(device)

    def set_rollout_model(self) -> None:
        """Choose which model(s) to use for rollout."""
        if self.rollout_mode == "random":
            self.rollout_index = random.randint(0, self.cfg.num_models - 1)

    def set_rollout_context(
        self,
        mu_in: torch.Tensor,
        std_in: torch.Tensor,
        mu_out: torch.Tensor,
        std_out: torch.Tensor,
    ) -> None:
        """Convenience method to set BOTH normalizer and rollout model at once.

        You should call this before starting a new rollout phase.
        """
        self.set_normalizer(mu_in, std_in, mu_out, std_out)
        self.set_rollout_model()

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
        device = state.device

        # Lazy default normalizer if not set (identity stats).
        if self.mu_in is None:
            in_dim = self.state_dim + self.action_dim
            out_dim = self.state_dim + 1
            self.set_normalizer(
                mu_in=torch.zeros(in_dim, device=device),
                std_in=torch.ones(in_dim, device=device),
                mu_out=torch.zeros(out_dim, device=device),
                std_out=torch.ones(out_dim, device=device),
            )

        # Type narrowing for normalizer tensors
        assert (
            self.mu_in is not None
            and self.std_in is not None
            and self.mu_out is not None
            and self.std_out is not None
        )

        x = torch.cat([state, action], dim=-1)
        x_norm = (x - self.mu_in) / self.std_in
        s_norm, a_norm = x_norm[:, : self.state_dim], x_norm[:, self.state_dim :]

        # Select model outputs according to rollout_mode
        if self.rollout_mode == "random":
            mean, log_std, done_logit = self.models[self.rollout_index](s_norm, a_norm)
        elif self.rollout_mode == "mean":
            outs = [m(s_norm, a_norm) for m in self.models]
            means = torch.stack([o[0] for o in outs], dim=0).mean(0)
            log_stds = torch.stack([o[1] for o in outs], dim=0).mean(0)
            done_logits = torch.stack([o[2] for o in outs], dim=0).mean(0)
            mean, log_std, done_logit = means, log_stds, done_logits
        else:
            raise ValueError(f"Unknown rollout_mode: {self.rollout_mode}")

        # Stabilize log_std and sample/mean in normalized space
        log_std = torch.clamp(log_std, self.cfg.log_std_bounds[0], self.cfg.log_std_bounds[1])
        if not deterministic:
            y_norm = Normal(mean, log_std.exp()).rsample()
        else:
            y_norm = mean

        # Denormalize [Δs, r]
        y = y_norm * self.std_out + self.mu_out
        delta_s = y[:, : self.state_dim]
        reward = y[:, self.state_dim : self.state_dim + 1]

        next_state = state + delta_s
        done = (torch.sigmoid(done_logit) > self.cfg.done_threshold).to(torch.bool)
        return next_state, reward, done

    def train(self, dataloader: DataLoader[dict[str, torch.Tensor]]) -> dict[str, list[float]]:
        """Train ALL models using Gaussian NLL for [Δs,r] and BCE for done.

        - Computes z-score stats from the real dataset and sets them.
        - Supports validation split and early stopping.
        - If an optimizer is provided, it should include ALL models' params; otherwise, one is created.

        Returns a dict of training/validation losses per epoch (averaged across models).
        """
        raise NotImplementedError("Not implemented")

    def generate_rollouts(self, real_data: Experience, rollout_len: int) -> Experience:
        """Generate rollouts."""
        raise NotImplementedError("Not implemented")
