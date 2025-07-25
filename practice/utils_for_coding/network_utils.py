from collections.abc import Sequence
from typing import cast

import torch
import torch.nn as nn


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
