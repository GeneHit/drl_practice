from collections.abc import Sequence
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from practice.utils.dist_utils import unwrap_model


def init_weights(layer: nn.Module) -> None:
    """
    Initialize weights for Conv2d and Linear layers using orthogonal initialization
    with appropriate gain for ReLU activations, and zero biases.
    """
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        # Use orthogonal initialization
        nn.init.orthogonal_(layer.weight, gain=int(nn.init.calculate_gain("relu")))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def load_checkpoint_if_exists(model: nn.Module, checkpoint_pathname: str) -> None:
    """Load a checkpoint into a model.

    Args:
        model: The model to load the checkpoint into.
        checkpoint_path: The path to the checkpoint.
    """
    if not checkpoint_pathname:
        return

    checkpoint = torch.load(checkpoint_pathname, weights_only=False)
    if isinstance(checkpoint, dict):
        # It's a state_dict
        model.load_state_dict(checkpoint)
    else:
        # It's a full model, extract state_dict
        model.load_state_dict(checkpoint.state_dict())


def soft_update(source: nn.Module, target: nn.Module, tau: torch.Tensor | float) -> None:
    """Soft update a target network.

    Args:
        source: The source network.
        target: The target network.
        tau: The soft update factor.
    """
    for param, target_param in zip(source.parameters(), target.parameters()):
        # equivalent to: tau * param + (1 - tau) * target_param
        target_param.data.lerp_(param.data, tau)


def save_model(net: nn.Module, pathname: str, full_model: bool = False) -> None:
    """Save the model."""
    model = unwrap_model(net)
    if full_model:
        torch.save(model, pathname)
    else:
        # save the model as a state dict
        torch.save(model.state_dict(), pathname)


def load_model(pathname: str, device: torch.device, net: nn.Module | None = None) -> nn.Module:
    """Load the model.

    Args:
        pathname: The path to the model.
        device: The device to load the model to.
        net: The network to load the model into. If None, will load a full model.

    Returns:
        The loaded model.
    """
    if net is None:
        # load a full model
        return cast(
            nn.Module, torch.load(pathname, map_location=device, weights_only=False).to(device)
        )

    # load the state dict
    net = net.to(device)
    load_checkpoint_if_exists(net, pathname)
    return net


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
        output_activation: type[nn.Module] | None = None,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(activation())
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())

        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x))


class DoubleQCritic(nn.Module):
    """Critic is a function that takes a state and an action and returns a Q-value.

    Have 2 Q-networks to reduce overestimation bias.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        input_dim = state_dim + action_dim

        self.q1 = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=nn.ReLU,
            use_layer_norm=use_layer_norm,
        )
        self.q2 = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=nn.ReLU,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        q1_val = self.q1(sa)
        q2_val = self.q2(sa)
        return q1_val, q2_val


class LogStdHead(nn.Module):
    """LogStdHead is a head that outputs the logstd of the action distribution.

    Support:
    1. state-dependent logstd with warm-start
    2. constant logstd
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dependent: bool = False,
        init_log_std: float = 1.0,
        log_std_min: float = -20,
        log_std_max: float = 2,
        map_to_range: bool = False,
    ) -> None:
        """
        Args:
            input_dim: Last feature dim (ignored if state_dependent=False)
            output_dim: Action dim
            state_dependent: If True, use nn.Linear; else use nn.Parameter
            init_log_std: Initial value for log_std (whether bias or constant)
            log_std_min, log_std_max: Clamp range for log_std
            map_to_range: Whether to map log_std to [log_std_min, log_std_max] for stability when
                state_dependent=True.
        """
        super().__init__()
        self.state_dependent = state_dependent
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.map_to_range = map_to_range

        if state_dependent:
            self.linear = nn.Linear(input_dim, output_dim)
            nn.init.zeros_(self.linear.weight)
            nn.init.constant_(self.linear.bias, init_log_std)
            self.forward_impl = self._state_dependent_forward
        else:
            self.log_std_param = nn.Parameter(torch.ones(output_dim) * init_log_std)
            self.forward_impl = self._constant_forward

    def forward(self, feature: Tensor, mean: Tensor) -> Tensor:
        """
        Args:
            feature: Feature vector from shared MLP [batch_size, input_dim]
            mean: Mean of action distribution [batch_size, act_dim]

        Returns:
            log_std: Tensor of shape [batch_size, act_dim]
        """
        log_std = self.forward_impl(feature, mean)
        return torch.clamp(log_std, self.log_std_min, self.log_std_max)

    def _state_dependent_forward(self, feature: Tensor, mean: Tensor) -> Tensor:
        log_std = self.linear(feature)
        if self.map_to_range:
            log_std = torch.tanh(log_std)
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (1 + log_std)
        return cast(Tensor, log_std)

    def _constant_forward(self, feature: Tensor, mean: Tensor) -> Tensor:
        return self.log_std_param.expand_as(mean)
