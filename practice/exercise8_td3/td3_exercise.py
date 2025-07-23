from typing import cast

import torch
from numpy.typing import NDArray
from torch import nn

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import ActTypeC, ObsType
from practice.base.trainer import TrainerBase
from practice.utils_for_coding.network_utils import init_weights
from practice.utils_for_coding.replay_buffer_utils import Experience


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
        output_activation: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(activation())
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())

        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x))


class ContinuousActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=nn.ReLU,
            output_activation=nn.Tanh,  # for scaling output to [-1,1]
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(state) * self.max_action)


class TD3Critic(nn.Module):
    """Critic is a function that takes a state and an action and returns a Q-value.

    Have 2 Q-networks to reduce overestimation bias.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        input_dim = state_dim + action_dim

        self.q1 = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=nn.ReLU,
        )
        self.q2 = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=nn.ReLU,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        q1_val = self.q1(sa)
        q2_val = self.q2(sa)
        return q1_val, q2_val


class TD3Config(BaseConfig):
    """The configuration for the TD3 algorithm."""

    total_steps: int
    """The step number to train the policy.

    The total step data is timesteps = total_steps / vector_env_num.
    """
    hidden_sizes: tuple[int, ...]
    """The hidden sizes of the MLP."""
    critic_lr: float
    """The learning rate for the critic."""
    replay_buffer_capacity: int
    """The capacity of the replay buffer."""
    batch_size: int
    """The batch size for training."""
    update_start_step: int
    """The step to start updating the critic and actor."""
    policy_delay: int
    """The step interval for updating the policy.

    The policy is updated every `policy_delay` critic updates.
    """
    policy_noise: float = 0.2
    """The standard deviation of the noise for the policy. Default Gaussian noise."""
    noise_clip: float = 0.5
    """The clip value of the noise for the policy."""
    exploration_noise: float = 0.1
    """The exploration noise for the policy."""
    max_action: float
    """The maximum action value."""
    tau: float = 0.005
    """The soft update factor for the target networks."""


class TD3Context(ContextBase):
    """The context for the TD3 algorithm."""

    critic: nn.Module
    """The critic network."""
    critic_optimizer: torch.optim.Optimizer
    """The optimizer for the critic."""


class TD3Trainer(TrainerBase):
    """The trainer for the TD3 algorithm."""

    def __init__(self, config: TD3Config, ctx: TD3Context) -> None:
        super().__init__(config, ctx)

        self._config: TD3Config = config
        self._ctx: TD3Context = ctx

    def train(self) -> None:
        """Train the TD3 algorithm.

        Steps:
        1. initialization: replay buffer, _TD3Pod, reset
        2. loop:
            - interact with environment
            - collect valid data
            - sample batch
            - update:
                - update critic
                - update actor if necessary
                - update target networks (soft update) if necessary
        """
        pass


class _TD3Pod:
    """The TD3 pod for training."""

    def __init__(self, config: TD3Config, ctx: TD3Context) -> None:
        self._config: TD3Config = config
        self._ctx: TD3Context = ctx

    def action(self, state: NDArray[ObsType], step: int) -> NDArray[ActTypeC]:
        """Get the action for the given state.

        Parameters
        ----------
        state : NDArray[ObsType]
        """
        raise NotImplementedError("Not implemented")

    def update(self, experience: Experience) -> None:
        """Update the TD3 pod.

        Steps:
        1. update critic
        2. update actor if necessary
        3. update target networks (soft update) if necessary
        """
        raise NotImplementedError("Not implemented")
