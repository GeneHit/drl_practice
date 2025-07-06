"""Train the DQN agent.

Reference:
Algorithm: https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm
Code:https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
"""

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter
from common.base import PolicyBase


class QNetwork(nn.Module):
    """Q network."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x)
        assert isinstance(y, torch.Tensor)  # make mypy happy
        return y


class DQNAgent(PolicyBase):
    """DQN agent."""

    def __init__(
        self,
        q_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        action_n: int,
    ) -> None:
        self._q_network = q_network
        self._optimizer = optimizer
        self._train_flag = False
        self._device = next(q_network.parameters()).device
        self._action_n = action_n

    @property
    def q_network(self) -> nn.Module:
        return self._q_network

    def set_train_flag(self, train_flag: bool) -> None:
        self._train_flag = train_flag
        self._q_network.train(train_flag)

    def action(self, state: Any, epsilon: float | None = None) -> int:
        if self._train_flag:
            assert epsilon is not None, "Epsilon is required in training mode"
            if random.random() < epsilon:
                # Exploration: take a random action with probability epsilon.
                return int(random.randint(0, self._action_n - 1))

        # 2 case:
        # >> 1. need exploitation: take the action with the highest value.
        # >> 2. in the test phase, take the action with the highest value.
        assert isinstance(state, np.ndarray), "State must be a numpy array"
        state_tensor = self._change_state_to_tensor(state).to(self._device)
        probs = self._q_network(state_tensor).cpu()
        return int(probs.argmax().item())

    def get_score(self, state: Any, action: int | None = None) -> float:
        assert isinstance(state, np.ndarray), "State must be a numpy array"
        state_tensor = self._change_state_to_tensor(state).to(self._device)
        probs = self._q_network(state_tensor).cpu()
        if action is None:
            return float(probs.max().item())
        return float(probs[0, action].item())

    def update(
        self, state: Any | None, action: Any | None, reward_target: Any
    ) -> None:
        assert isinstance(state, np.ndarray), "State must be a numpy array"
        assert isinstance(action, np.ndarray), "Action must be a numpy array"
        assert isinstance(reward_target, torch.Tensor), "Score must be a tensor"
        assert reward_target.dim() == 1, "Score must be a 1D tensor"

        state_tensor = self._change_state_to_tensor(state).to(self._device)
        # dimention: [batch_size, 1]
        actions = torch.from_numpy(action).view(-1, 1).to(self._device)
        # [batch_size, num_actions] -gather-> [batch_size, 1]
        # old_values: [batch_size, 1]
        old_values = self._q_network(state_tensor).gather(1, actions).squeeze()
        td_target = reward_target.to(self._device)

        # optimize the model
        self._optimizer.zero_grad()
        loss = F.mse_loss(old_values, td_target)
        loss.backward()
        self._optimizer.step()

    def save(self, pathname: str) -> None:
        """Save the DQN model."""
        # only save the q_network
        torch.save(self._q_network.state_dict(), pathname)

    @classmethod
    def load(cls, pathname: str) -> "DQNAgent":
        """Load the DQN model."""
        raise NotImplementedError(
            "load the QNetwork outside of the DQNAgent class"
        )

    def _change_state_to_tensor(self, state: Any) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            if state.ndim == 1:
                # Fix numpy array view call - use reshape instead of view
                state = state.reshape(1, -1)
            return torch.tensor(state, dtype=torch.float32)
        elif isinstance(state, torch.Tensor):
            return state
        else:
            raise ValueError("Invalid state type")
