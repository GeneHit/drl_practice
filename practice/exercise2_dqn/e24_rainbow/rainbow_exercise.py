import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Discrete
from numpy.typing import NDArray

from practice.base.context import ContextBase
from practice.base.env_typing import ActType, ObsType
from practice.exercise2_dqn.dqn_exercise import DQNConfig, DQNPod, get_dqn_actions
from practice.exercise2_dqn.e24_rainbow.per_exercise import PERBuffer, PERBufferConfig
from practice.utils_for_coding.network_utils import MLP, init_weights
from practice.utils_for_coding.numpy_tensor_utils import argmax_action
from practice.utils_for_coding.writer_utils import CustomWriter


@dataclass(kw_only=True, frozen=True)
class RainbowConfig(DQNConfig):
    """Rainbow DQN Config."""

    noisy_std: float
    """The standard deviation of the noisy layer."""
    per_buffer_config: PERBufferConfig
    """The configuration for the prioritized experience replay buffer."""


class NoisyLinear(nn.Module):
    """Noisy Linear Layer.

    Reference:
    - https://arxiv.org/abs/1706.10295
    """

    def __init__(self, in_features: int, out_features: int, std: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std = std
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / std)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented")


class RainbowNet(nn.Module):
    """Rainbow Network.

    Rainbow network is a combination of:
    - dueling: https://arxiv.org/abs/1511.06581
    - noisy: https://arxiv.org/abs/1706.10295
    - distributional: https://arxiv.org/abs/1707.06887
    """

    def __init__(
        self,
        state_n: int,
        action_n: int,
        hidden_sizes: tuple[int, ...],
        noisy_std: float,
    ) -> None:
        super().__init__()
        # feature stream
        self.feature = MLP(
            input_dim=state_n,
            hidden_sizes=hidden_sizes,
            output_dim=action_n,
        )
        # value stream
        self.value_head = nn.Sequential(nn.Linear(hidden_sizes[-1], 1))
        # advantage stream
        self.advantage_head = nn.Sequential(nn.Linear(hidden_sizes[-1], action_n))
        self.advantage_head.apply(init_weights)
        self.value_head.apply(init_weights)
        # noisy layer
        self.noisy_layer = NoisyLinear(hidden_sizes[-1], action_n, std=noisy_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented")

    def action(self, x: torch.Tensor) -> ActType:
        """Get the action for evaluation/gameplay with 1 environment.

        Returns:
            action: The single action.
        """
        # greedy strategy
        return argmax_action(self.forward(x), dtype=ActType)


class RainbowPod(DQNPod):
    """Rainbow DQN Pod.

    Rainbow DQN is a combination of:
    - Double DQN: https://arxiv.org/abs/1509.06461
    - Dueling DQN: https://arxiv.org/abs/1511.06581
    - Noisy DQN: https://arxiv.org/abs/1706.10295
    - Distributional DQN: https://arxiv.org/abs/1707.06887 (C51)
    - Prioritized Experience Replay Buffer: https://arxiv.org/abs/1511.05952 (PER)

    Reference:
    - https://arxiv.org/abs/1710.02298
    """

    def __init__(self, config: DQNConfig, ctx: ContextBase, writer: CustomWriter) -> None:
        # use DQNConfig for the base class
        assert isinstance(config, RainbowConfig)
        self._config: RainbowConfig = config
        self._ctx = ctx
        self._writer = writer
        self._noisy_std = config.noisy_std
        self._replay_buffer = PERBuffer(config.per_buffer_config)
        self._target_net = copy.deepcopy(ctx.network)

        self._target_net.eval()
        self._ctx.network.train()

        # Get action space info
        assert isinstance(self._ctx.eval_env.action_space, Discrete)
        self._action_n = int(self._ctx.eval_env.action_space.n)
        self._step = 0

    def sync_target_net(self) -> None:
        self._target_net.load_state_dict(self._ctx.network.state_dict())

    def action(self, states: NDArray[ObsType]) -> NDArray[ActType]:
        actions = get_dqn_actions(
            network=self._ctx.network,
            states=states,
            epsilon=self._config.epsilon_schedule(self._step),
            env_state_shape=self._ctx.env_state_shape,
            action_n=self._action_n,
            step=self._step,
            writer=self._writer,
            log_interval=self._config.log_interval,
            device=self._config.device,
        )
        self._step += 1
        return actions

    def update(self) -> None:
        raise NotImplementedError("Not implemented")

    def buffer_add(
        self,
        states: NDArray[ObsType],
        actions: NDArray[ActType],
        rewards: NDArray[np.float32],
        next_states: NDArray[ObsType],
        dones: NDArray[np.bool_],
        env_idxs: NDArray[np.int16],
    ) -> None:
        self._replay_buffer.add_batch(states, actions, rewards, next_states, dones, env_idxs)
