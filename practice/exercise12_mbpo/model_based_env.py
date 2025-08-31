import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from practice.utils_for_coding.buffer import Experience, ReplayBuffer
from practice.utils_for_coding.network_utils import MLP


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Training config for dynamics models."""

    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    loss_weight_delta: float = 1.0  # weight for Δs NLL
    loss_weight_reward: float = 1.0  # weight for r NLL
    loss_weight_done: float = 1.0  # weight for BCE(done)
    bootstrap: bool = True

    buffer_ratio_for_val: float = 0.1
    """The ratio of the buffer to sample from for validation."""
    early_stop_patience: int = 10
    """The patience for early stopping."""

    dataloader_num_workers: int = 0
    """The number of workers for the dataloader."""
    dataloader_pin_memory: bool = False
    """Whether to pin the memory for the dataloader."""


@dataclass(frozen=True, kw_only=True)
class ModelBasedConfig:
    """Config consumed by ModelBasedEnv (includes TrainConfig)."""

    num_models: int
    model_hidden_sizes: tuple[int, ...]
    done_threshold: float = 0.5
    log_std_bounds: tuple[float, float] = (-5.0, 2.0)
    eps: float = 1e-6
    train: TrainConfig


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
        self._models = [model, *[copy.deepcopy(model) for _ in range(cfg.num_models - 1)]]
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
        self._mu_in: Optional[Tensor] = None  # shape (state_dim + action_dim,)
        self._std_in: Optional[Tensor] = None
        self._mu_out: Optional[Tensor] = None  # shape (state_dim + 1,) for [Δs, r]
        self._std_out: Optional[Tensor] = None

        # Rollout selector
        self._rollout_model_index: int = 0
        self._device = next(self._models[0].parameters()).device

    def set_rollout_model(self) -> None:
        """Choose which model(s) to use for rollout."""
        self._rollout_model_index = random.randint(0, self._cfg.num_models - 1)

    @torch.no_grad()
    def step(
        self, state: Tensor, action: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict (next_state, reward, done) using the selected rollout model(s).

        Args:
            state:  (B, state_dim)
            action: (B, action_dim)
            deterministic: whether to sample or use mean for rollout
                if True, use mean for rollout
                if False, sample from the distribution

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

    def _get_and_set_normalizer(self, buffer: ReplayBuffer) -> None:
        """Get and set z-score stats for inputs [s,a] and outputs [Δs,r]."""
        # get all data from buffer
        exp = buffer.sample(len(buffer))

        # calculate mean and std for inputs [s,a]
        inputs = torch.cat([exp.states, exp.actions], dim=-1)
        mu_in = inputs.mean(dim=0)
        std_in = inputs.std(dim=0, unbiased=False)

        # calculate mean and std for outputs [Δs,r]
        delta_states = exp.next_states - exp.states
        outputs = torch.cat([delta_states, exp.rewards.unsqueeze(-1)], dim=-1)
        mu_out = outputs.mean(dim=0)
        std_out = outputs.std(dim=0, unbiased=False)

        # Set z-score stats for inputs [s,a] and outputs [Δs,r]
        self._mu_in = mu_in.to(self._device)
        self._std_in = std_in.clamp_min(self._cfg.eps).to(self._device)
        self._mu_out = mu_out.to(self._device)
        self._std_out = std_out.clamp_min(self._cfg.eps).to(self._device)

    def _gauss_nll(self, mean: Tensor, log_std: Tensor, target: Tensor) -> Tensor:
        """Diagonal Gaussian NLL (mean over batch & dims) in normalized space."""
        log_std = torch.clamp(log_std, self._cfg.log_std_bounds[0], self._cfg.log_std_bounds[1])
        var = (log_std.exp()) ** 2
        nll = 0.5 * (((target - mean) ** 2) / var + 2.0 * log_std)
        return nll.mean()

    def _normalize_io(self, exp: Experience) -> tuple[Tensor, Tensor, Tensor]:
        """Return (s_norm, a_norm, y_norm) where y_norm is normalized [Δs, r]."""
        assert (
            self._mu_in is not None
            and self._std_in is not None
            and self._mu_out is not None
            and self._std_out is not None
        )

        x = torch.cat([exp.states, exp.actions], dim=-1)
        x_norm = (x - self._mu_in) / self._std_in
        s_norm = x_norm[:, : self._state_dim]
        a_norm = x_norm[:, self._state_dim :]

        delta_s = exp.next_states - exp.states
        y = torch.cat([delta_s, exp.rewards.unsqueeze(-1)], dim=-1)
        y_norm = (y - self._mu_out) / self._std_out
        return s_norm, a_norm, y_norm

    @torch.no_grad()
    def _validate_epoch(self, val_exp: Experience) -> tuple[float, float, float, float]:
        """Return (total, delta, reward, done) validation loss averages (averaged across models)."""
        s_norm, a_norm, y_norm = self._normalize_io(val_exp)

        delta_t = y_norm[:, : self._state_dim]
        reward_t = y_norm[:, self._state_dim : self._state_dim + 1]

        bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

        totals = []
        deltas = []
        rewards = []
        dones = []
        for model in self._models:
            model.eval()
            mean, log_std, done_logit = model(s_norm, a_norm)

            mean_d = mean[:, : self._state_dim]
            mean_r = mean[:, self._state_dim : self._state_dim + 1]
            log_d = log_std[:, : self._state_dim]
            log_r = log_std[:, self._state_dim : self._state_dim + 1]

            loss_delta = self._gauss_nll(mean_d, log_d, delta_t)
            loss_reward = self._gauss_nll(mean_r, log_r, reward_t)
            loss_done = bce(done_logit, val_exp.dones.float().unsqueeze(-1))
            loss_total = (
                self._cfg.train.loss_weight_delta * loss_delta
                + self._cfg.train.loss_weight_reward * loss_reward
                + self._cfg.train.loss_weight_done * loss_done
            )

            totals.append(loss_total.item())
            deltas.append(loss_delta.item())
            rewards.append(loss_reward.item())
            dones.append(loss_done.item())

        return (
            float(torch.tensor(totals).mean().item()),
            float(torch.tensor(deltas).mean().item()),
            float(torch.tensor(rewards).mean().item()),
            float(torch.tensor(dones).mean().item()),
        )

    def _train_one_batch(
        self, model: nn.Module, opt: torch.optim.Optimizer, exp: Experience
    ) -> tuple[float, float, float, float]:
        """Train one model on one batch, return (total, delta, reward, done) loss scalars."""
        model.train()

        s_norm, a_norm, y_norm = self._normalize_io(exp)
        delta_t = y_norm[:, : self._state_dim]
        reward_t = y_norm[:, self._state_dim : self._state_dim + 1]

        mean, log_std, done_logit = model(s_norm, a_norm)

        mean_d = mean[:, : self._state_dim]
        mean_r = mean[:, self._state_dim : self._state_dim + 1]
        log_d = log_std[:, : self._state_dim]
        log_r = log_std[:, self._state_dim : self._state_dim + 1]

        loss_delta = self._gauss_nll(mean_d, log_d, delta_t)
        loss_reward = self._gauss_nll(mean_r, log_r, reward_t)
        bce = torch.nn.BCEWithLogitsLoss(reduction="mean")
        loss_done = bce(done_logit, exp.dones.float().unsqueeze(-1))

        loss_total = (
            self._cfg.train.loss_weight_delta * loss_delta
            + self._cfg.train.loss_weight_reward * loss_reward
            + self._cfg.train.loss_weight_done * loss_done
        )

        opt.zero_grad(set_to_none=True)
        loss_total.backward()
        opt.step()

        return (loss_total.item(), loss_delta.item(), loss_reward.item(), loss_done.item())

    def train(self, buffer: ReplayBuffer) -> dict[str, list[float]]:
        """Train ALL models using Gaussian NLL for [Δs,r] and BCE for done.

        - Computes z-score stats from the real dataset and sets them.
        - Supports validation split and early stopping.
        - If an optimizer is provided, it should include ALL models' params; otherwise, one is created.

        Returns a dict of training/validation losses per epoch (averaged across models).
        """
        # set normalizer for later rollout generation (call step() after training)
        self._get_and_set_normalizer(buffer)

        # build val split
        val_exp = _build_val_split(buffer, self._cfg.train.buffer_ratio_for_val).to(self._device)

        # history
        hist: dict[str, list[float]] = {
            "train_total": [],
            "train_delta": [],
            "train_reward": [],
            "train_done": [],
            "val_total": [],
            "val_delta": [],
            "val_reward": [],
            "val_done": [],
        }

        # early stopping state
        best_val = float("inf")
        best_states = [copy.deepcopy(m.state_dict()) for m in self._models]
        bad_epochs = 0
        patience = self._cfg.train.early_stop_patience

        # train multiple epochs
        for _ in range(self._cfg.train.epochs):
            loader = buffer.dataloader(
                self._cfg.train.batch_size,
                ratio=1 - self._cfg.train.buffer_ratio_for_val,
                shuffle=True,
                num_workers=self._cfg.train.dataloader_num_workers,
                pin_memory=self._cfg.train.dataloader_pin_memory,
            )

            total_list, delta_list, reward_list, done_list = [], [], [], []

            for batch in loader:
                exp = Experience.from_kwargs(**batch).to(self._device)

                for model, opt in zip(self._models, self._optimizers):
                    exp_m = _bootstrap_exp(exp) if self._cfg.train.bootstrap else exp
                    t, dlt, rwd, dn = self._train_one_batch(model, opt, exp_m)
                    total_list.append(t)
                    delta_list.append(dlt)
                    reward_list.append(rwd)
                    done_list.append(dn)

            # record training mean
            hist["train_total"].append(float(torch.tensor(total_list).mean().item()))
            hist["train_delta"].append(float(torch.tensor(delta_list).mean().item()))
            hist["train_reward"].append(float(torch.tensor(reward_list).mean().item()))
            hist["train_done"].append(float(torch.tensor(done_list).mean().item()))

            # validate
            vt, vd, vr, vdn = self._validate_epoch(val_exp)
            hist["val_total"].append(vt)
            hist["val_delta"].append(vd)
            hist["val_reward"].append(vr)
            hist["val_done"].append(vdn)

            # early stopping
            current_val = vt
            if not (current_val != current_val):  # filter NaN
                if current_val + 1e-8 < best_val:
                    best_val = current_val
                    best_states = [copy.deepcopy(m.state_dict()) for m in self._models]
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if bad_epochs >= patience:
                    for m, sd in zip(self._models, best_states):
                        m.load_state_dict(sd)
                    return hist

        # roll back to best
        if best_val < float("inf"):
            for m, sd in zip(self._models, best_states):
                m.load_state_dict(sd)

        return hist


def _build_val_split(buffer: ReplayBuffer, val_ratio: float) -> Experience:
    """Build a validation set tensor.

    Args:
        buffer: The buffer to sample from.
        val_ratio: The ratio of the buffer to sample from for validation. It will use the
            [1-val_ratio, 1] part of the buffer for validation.

    Returns:
        The validation set.
    """
    N = len(buffer)
    n_val = max(1, int(N * val_ratio)) if N > 1 else 0
    assert n_val > 0
    idxs = torch.randint(N - n_val, N, (n_val,))
    return buffer.sample_by_idxs(idxs)


def _bootstrap_exp(exp: Experience) -> Experience:
    n = exp.states.shape[0]
    idx = torch.randint(0, n, (n,), device=exp.states.device)
    return Experience(
        states=exp.states.index_select(0, idx),
        actions=exp.actions.index_select(0, idx),
        rewards=exp.rewards.index_select(0, idx),
        next_states=exp.next_states.index_select(0, idx),
        dones=exp.dones.index_select(0, idx),
    )
