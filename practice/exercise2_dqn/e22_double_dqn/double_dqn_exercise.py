import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ObsType
from practice.exercise2_dqn.dqn_exercise import BasicDQNPod, DQNConfig
from practice.utils_for_coding.writer_utils import CustomWriter


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
        env_idxs: NDArray[np.int16],
    ) -> None:
        """Add batch of experiences to the replay buffer."""
        super().buffer_add(states, actions, rewards, next_states, dones, env_idxs)

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
            data={"loss/td_loss": loss},
            step=self._step,
            log_interval=self._config.log_interval,
            blocked=False,
        )
