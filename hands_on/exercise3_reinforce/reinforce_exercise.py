from typing import Sequence, TypeAlias, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

# from torch.utils.tensorboard import SummaryWriter
from hands_on.base import ActType
from hands_on.exercise3_reinforce.config import ReinforceConfig

ObsType: TypeAlias = Union[np.uint8, np.float32]
ArrayType: TypeAlias = Union[np.bool_, np.float32]
EnvType: TypeAlias = gym.Env[NDArray[ObsType], ActType]
EnvsType: TypeAlias = gym.vector.VectorEnv[
    NDArray[ObsType], ActType, NDArray[ArrayType]
]


class Reinforce1DNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        layer_num: int = 2,
    ) -> None:
        super().__init__()
        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        ]

        for _ in range(layer_num - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = self.network(x)  # make mypy happy
        return y


class ReinforceTrainer:
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gamma: float,
        state_shape: tuple[int, ...],
        grad_acc: int = 1,
    ) -> None:
        """Initialize the trainer.

        Args:
            net: The policy network.
            optimizer: The optimizer.
            device: The device to run the network.
            gamma: The discount factor.
            state_shape: The shape of the state.
            grad_acc: The number of gradient accumulation.
        """
        self._net = net
        self._optimizer = optimizer
        self._device = device
        self._gamma = gamma
        self._state_shape = state_shape
        self._grad_acc = grad_acc

    def action(
        self, state: NDArray[ObsType], eval: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Get action(s) and log_prob(s) for state(s).

        Args:
            state: Single state or batch of states

        Returns:
            actions: Tensor
                Single action or batch of actions depending on input shape.
                If the input is a single state, output a (1, ) array, making
                the output type consistent.
            log_probs: NDArray[np.float32]
                Log probability of the actions.
        """
        is_single = len(state.shape) == len(self._state_shape)
        state_batch = state if not is_single else state.reshape(1, *state.shape)
        state_tensor = torch.from_numpy(state_batch).to(self._device)

        # if train, sample from categorical distribution
        probs = self._net(state_tensor).cpu()
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs

    def update(
        self, rewards: Sequence[float], log_probs: Sequence[Tensor]
    ) -> None:
        """Update the policy network with a episode's data.

        Don't support batch episodes now, because it's complex to collect the
        multi-episode batch data:
        1. for single env, use gradient accumulation is a good way.
        2. for multi env, different env has different episode length.
            - so it's complex to collect the multi-episode batch data.
            - if use the same episode length, it's not efficient.

        Args:
            rewards: rewards of a episode
            log_probs: log probabilities of actions in a episode
        """
        pass


def reinforce_train_loop(
    envs: EnvsType,
    net: nn.Module,
    device: torch.device,
    config: ReinforceConfig,
) -> None:
    """Train the policy network with multiple environments."""
    raise NotImplementedError
