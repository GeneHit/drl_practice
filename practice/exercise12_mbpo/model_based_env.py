import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sized, cast

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataset import Subset

from practice.utils_for_coding.network_utils import MLP


@dataclass
class TrainConfig:
    """Training config for dynamics models."""

    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    loss_weight_delta: float = 1.0  # weight for Δs NLL
    loss_weight_reward: float = 1.0  # weight for r NLL
    loss_weight_done: float = 1.0  # weight for BCE(done)
    val_ratio: float = 0.1
    early_stop_patience: int = 10
    bootstrap: bool = True  # bootstrap per model (with replacement)


@dataclass
class ModelBasedConfig:
    """Config consumed by ModelBasedEnv (includes TrainConfig)."""

    train: TrainConfig = TrainConfig()
    done_threshold: float = 0.5
    log_std_bounds: tuple[float, float] = (-5.0, 2.0)
    eps: float = 1e-6
    rollout_mode: Literal["random", "mean"] = "random"


class EnvModel(nn.Module):
    """Predict the [Δs,r], and done's logit.

    Output dimensions:
      mean:     (B, state_dim+1) for [Δs, r]
      log_std:  (B, state_dim+1)
      done_logit: (B, 1)
    """

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
        x = torch.cat([state, action], dim=-1)
        h = self.backbone(x)
        gauss = self.head_gauss(h)
        mean, log_std = torch.chunk(gauss, 2, dim=-1)
        done_logit = self.head_done(h)
        return mean, log_std, done_logit


class ModelBasedEnv:
    """A model-based environment wrapper that holds an ensemble (list) of dynamics models."""

    def __init__(self, models: list[nn.Module], cfg: ModelBasedConfig) -> None:
        assert len(models) > 0, "At least one model is required."
        self.models: list[nn.Module] = models
        self.cfg = cfg

        # Infer state/action dims from the first model (all models should match).
        m0 = self.models[0]
        assert hasattr(m0, "state_dim") and hasattr(m0, "action_dim"), (
            "Each model must expose state_dim and action_dim."
        )
        m0_any = cast(Any, m0)
        self.state_dim = int(m0_any.state_dim)
        self.action_dim = int(m0_any.action_dim)

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
            self.rollout_index = random.randint(0, len(self.models) - 1)

    def set_rollout_context(
        self,
        mu_in: torch.Tensor,
        std_in: torch.Tensor,
        mu_out: torch.Tensor,
        std_out: torch.Tensor,
    ) -> None:
        """
        Convenience method to set BOTH normalizer and rollout model at once.
        This is often what you want when starting a new rollout phase.
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
        done_prob = torch.sigmoid(done_logit)
        done = (done_prob > self.cfg.done_threshold).to(next_state.dtype)
        return next_state, reward, done

    def train(
        self,
        states: torch.Tensor,  # (N, state_dim)
        actions: torch.Tensor,  # (N, action_dim)
        rewards: torch.Tensor,  # (N, 1)
        next_states: torch.Tensor,  # (N, state_dim)
        dones: torch.Tensor,  # (N, 1)
        optimizer: Optional[torch.optim.Optimizer] = None,
        progress: bool = True,
    ) -> dict[str, list[float]]:
        """Train ALL models using Gaussian NLL for [Δs,r] and BCE for done.

        - Computes z-score stats from the real dataset and sets them.
        - Supports validation split and early stopping.
        - If an optimizer is provided, it should include ALL models' params; otherwise, one is created.
        Returns a dict of training/validation losses per epoch (averaged across models).
        """
        device = next(self.models[0].parameters()).device
        states, actions, rewards, next_states, dones = [
            t.to(device) for t in (states, actions, rewards, next_states, dones)
        ]
        N, sdim = states.shape
        assert sdim == self.state_dim, "state_dim mismatch."

        # 1) Compute z-score stats and cache them
        delta_states = next_states - states
        X_in = torch.cat([states, actions], dim=-1)  # (N, s+a)
        Y_out = torch.cat([delta_states, rewards], dim=-1)  # (N, s+1)
        mu_in, std_in = X_in.mean(0), X_in.std(0).clamp_min(self.cfg.eps)
        mu_out, std_out = Y_out.mean(0), Y_out.std(0).clamp_min(self.cfg.eps)
        self.set_normalizer(mu_in, std_in, mu_out, std_out)

        # Pre-normalize data for faster training
        assert (
            self.mu_in is not None
            and self.std_in is not None
            and self.mu_out is not None
            and self.std_out is not None
        )
        Xn = (X_in - self.mu_in) / self.std_in
        Yn = (Y_out - self.mu_out) / self.std_out

        full_ds: TorchDataset[tuple[torch.Tensor, ...]] = TensorDataset(
            Xn[:, :sdim],  # s_norm
            Xn[:, sdim:],  # a_norm
            Yn[:, :sdim],  # Δs_norm
            Yn[:, sdim:],  # r_norm
            dones,  # done (0/1)
        )
        # Validation split
        if self.cfg.train.val_ratio > 0.0 and N >= 10:
            n_val = max(1, int(N * self.cfg.train.val_ratio))
            n_train = N - n_val
            train_ds: TorchDataset[tuple[torch.Tensor, ...]]
            val_ds: Optional[TorchDataset[tuple[torch.Tensor, ...]]]
            train_ds, val_ds = random_split(
                full_ds, [n_train, n_val], generator=torch.Generator(device="cpu")
            )
        else:
            train_ds = full_ds
            val_ds = None

        # Optimizer
        if optimizer is None:
            params = []
            for m in self.models:
                params += list(m.parameters())
            optimizer = torch.optim.AdamW(
                params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay
            )

        def make_loader(
            ds: TorchDataset[tuple[torch.Tensor, ...]],
        ) -> DataLoader[tuple[torch.Tensor, ...]]:
            return DataLoader(
                ds, batch_size=self.cfg.train.batch_size, shuffle=True, drop_last=False
            )

        # Prepare per-model bootstrap loaders
        def bootstrap_indices(n_items: int) -> torch.Tensor:
            return torch.randint(0, n_items, (n_items,), device="cpu")

        train_len = len(cast(Sized, train_ds))
        base_train_indices = torch.arange(train_len)

        if val_ds is not None:
            val_len = len(cast(Sized, val_ds))
            base_val_indices = torch.arange(val_len)
        else:
            base_val_indices = None

        per_model_train_loaders: list[DataLoader[tuple[torch.Tensor, ...]]] = []
        per_model_val_loaders: list[Optional[DataLoader[tuple[torch.Tensor, ...]]]] = []
        for _ in range(len(self.models)):
            if self.cfg.train.bootstrap and not isinstance(train_ds, TensorDataset):
                idx = bootstrap_indices(len(base_train_indices))
                tr_subset: TorchDataset[tuple[torch.Tensor, ...]] = Subset(
                    train_ds, base_train_indices[idx].tolist()
                )
            else:
                if isinstance(train_ds, TensorDataset):
                    tr_subset = cast(TorchDataset[tuple[torch.Tensor, ...]], train_ds)
                else:
                    tr_subset = Subset(train_ds, base_train_indices.tolist())
            per_model_train_loaders.append(make_loader(tr_subset))

            if val_ds is not None:
                if base_val_indices is not None:
                    val_subset: TorchDataset[tuple[torch.Tensor, ...]] = Subset(
                        val_ds, base_val_indices.tolist()
                    )
                else:
                    val_subset = cast(TorchDataset[tuple[torch.Tensor, ...]], val_ds)
                per_model_val_loaders.append(make_loader(val_subset))
            else:
                per_model_val_loaders.append(None)

        # Loss helpers
        def gauss_nll(
            target: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
        ) -> torch.Tensor:
            log_std = torch.clamp(log_std, self.cfg.log_std_bounds[0], self.cfg.log_std_bounds[1])
            var = log_std.exp().pow(2).clamp_min(1e-12)
            return 0.5 * (((target - mean) ** 2) / var + 2 * log_std).sum(dim=1).mean()

        @torch.no_grad()
        def eval_model(
            m: nn.Module, loader: Optional[DataLoader[tuple[torch.Tensor, ...]]]
        ) -> float:
            if loader is None:
                return 0.0
            m.eval()
            acc = 0.0
            cnt = 0
            for sb, ab, dsn, rn, db in loader:
                mean, log_std, done_logit = m(sb, ab)
                loss = (
                    self.cfg.train.loss_weight_delta
                    * gauss_nll(dsn, mean[:, :sdim], log_std[:, :sdim])
                    + self.cfg.train.loss_weight_reward
                    * gauss_nll(rn, mean[:, sdim:], log_std[:, sdim:])
                    + self.cfg.train.loss_weight_done
                    * F.binary_cross_entropy_with_logits(done_logit, db)
                )
                acc += loss.item()
                cnt += 1
            return acc / max(1, cnt)

        logs: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = math.inf
        patience = self.cfg.train.early_stop_patience
        best_state: Optional[dict[str, torch.Tensor]] = None

        # Train for epochs (iterate each model per epoch)
        for ep in range(self.cfg.train.epochs):
            tl_list, vl_list = [], []
            for k, m in enumerate(self.models):
                m.train()
                train_loader = per_model_train_loaders[k]
                val_loader = (
                    per_model_val_loaders[k] if per_model_val_loaders[k] is not None else None
                )

                acc = 0.0
                cnt = 0
                for sb, ab, dsn, rn, db in train_loader:
                    mean, log_std, done_logit = m(sb, ab)
                    loss = (
                        self.cfg.train.loss_weight_delta
                        * gauss_nll(dsn, mean[:, :sdim], log_std[:, :sdim])
                        + self.cfg.train.loss_weight_reward
                        * gauss_nll(rn, mean[:, sdim:], log_std[:, sdim:])
                        + self.cfg.train.loss_weight_done
                        * F.binary_cross_entropy_with_logits(done_logit, db)
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    acc += loss.item()
                    cnt += 1
                tl = acc / max(1, cnt)
                vl = eval_model(m, val_loader) if val_loader is not None else tl
                tl_list.append(tl)
                vl_list.append(vl)

            tl_mean, vl_mean = (
                float(torch.tensor(tl_list).mean()),
                float(torch.tensor(vl_list).mean()),
            )
            logs["train_loss"].append(tl_mean)
            logs["val_loss"].append(vl_mean)
            if progress:
                print(
                    f"[Dynamics-Ensemble][epoch {ep + 1:03d}] train={tl_mean:.4f} val={vl_mean:.4f}"
                )

            # Early stopping on ensemble-mean val loss
            if vl_mean < best_val - 1e-6:
                best_val = vl_mean
                patience = self.cfg.train.early_stop_patience
                # snapshot all models
                best_state = {
                    f"m{k}.{n}": p.detach().cpu().clone()
                    for k, m in enumerate(self.models)
                    for n, p in m.state_dict().items()
                }
            else:
                patience -= 1
                if patience <= 0:
                    break

        # Load best snapshot
        if best_state is not None:
            with torch.no_grad():
                for k, m in enumerate(self.models):
                    sd = {
                        n.split(".", 1)[1]: best_state[f"m{k}.{n.split('.', 1)[1]}"]
                        for n in m.state_dict().keys()
                    }
                    m.load_state_dict(sd)

        return logs
