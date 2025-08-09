from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ObsType
from practice.exercise2_dqn.dqn_exercise import BasicDQNPod, DQNConfig
from practice.utils_for_coding.numpy_tensor_utils import argmax_action
from practice.utils_for_coding.writer_utils import CustomWriter


class QNet2D(nn.Module):
    """Q network with 2D convolution.

    Same as DeepMind's DQN paper for Atari:
    https://www.nature.com/articles/nature14236
    """

    def __init__(self, in_shape: tuple[int, int, int], action_n: int) -> None:
        super().__init__()
        c, h, w = in_shape
        # convolution layer sequence
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
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

        # full connected layer, original 512
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use cast to make mypy happy
        return cast(torch.Tensor, self.fc(self.conv(x / 255.0)))

    def action(self, x: torch.Tensor) -> ActType:
        """Get the action for evaluation/gameplay with 1 environment.

        Returns:
            action: The single action.
        """
        # greedy strategy
        return argmax_action(self.forward(x), dtype=ActType)


class DoubleDQNPod(BasicDQNPod):
    """Double DQN pod.

    The key difference from Basic DQN is in the target Q-value computation:
    - Basic DQN: uses target network for both action selection and evaluation
    - Double DQN: uses current network for action selection, target network for evaluation

    Reference:
    - https://arxiv.org/abs/1509.06461
    """

    def __init__(self, config: DQNConfig, ctx: ContextBase, writer: CustomWriter) -> None:
        # Notice: can use many variables from super class
        super().__init__(config, ctx, writer)

    def sync_target_net(self) -> None:
        """Synchronize target network with current Q-network."""
        super().sync_target_net()

    def buffer_add(
        self,
        states: NDArray[ObsType],
        actions: NDArray[ActType],
        rewards: NDArray[np.float32],
        next_states: NDArray[ObsType],
        dones: NDArray[np.bool_],
    ) -> None:
        """Add batch of experiences to the replay buffer."""
        super().buffer_add(states, actions, rewards, next_states, dones)

    def action(self, states: NDArray[ObsType]) -> NDArray[ActType]:
        """Get action(s) for state(s).

        Args:
            state: Single state or batch of states

        Returns:
            actions: NDArray[ActType]
                Single action or batch of actions depending on input shape.
        """
        return super().action(states)

    def update(self) -> None:
        """Update Q-network using experiences with Double DQN logic.

        Args:
            experiences: Batch of experiences from replay buffer

        Returns:
            loss: The TD loss value for logging
        """
        if len(self._replay) < self._config.batch_size:
            return

        # sample batch and move to device
        exps = self._replay.sample(self._config.batch_size).to(self._config.device)

        # Double DQN: use current network for action selection, target network for evaluation
        with torch.no_grad():
            # Use current network to select actions
            current_q_values = self._ctx.network(exps.next_states)
            next_actions = current_q_values.argmax(dim=1)

            # Use target network to evaluate the selected actions
            target_q_values = self._target_net(exps.next_states)
            target_max = target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()

            # Compute TD target
            td_target = exps.rewards.flatten() + self._config.gamma * target_max * (
                1 - exps.dones.flatten().float()
            )

        # Get current Q-values for the actions taken
        self._ctx.network.train()
        current_q = self._ctx.network(exps.states).gather(1, exps.actions.view(-1, 1)).squeeze()

        # Compute loss and update
        self._ctx.optimizer.zero_grad()
        loss = F.mse_loss(current_q, td_target)
        loss.backward()
        self._ctx.optimizer.step()

        self._writer.log_stats(
            data={"loss/td_loss": loss.item()},
            step=self._step,
            log_interval=self._config.log_interval,
            blocked=False,
        )
