"""Train the DQN agent.

Reference:
Algorithm: https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm
Code:https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
"""

from typing import TypeAlias, Union

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

ObsType: TypeAlias = Union[np.uint8, np.float32]
ArrayType: TypeAlias = Union[np.bool_, np.float32]

# D: discrete
ActType: TypeAlias = np.int64
EnvType: TypeAlias = gym.Env[NDArray[ObsType], ActType]
EnvsType: TypeAlias = gym.vector.VectorEnv[NDArray[ObsType], NDArray[ActType], NDArray[ArrayType]]

# C: continuous
ActTypeC: TypeAlias = np.float32
EnvTypeC: TypeAlias = gym.Env[NDArray[ObsType], ActTypeC]
EnvsTypeC: TypeAlias = gym.vector.VectorEnv[NDArray[ObsType], NDArray[ActTypeC], NDArray[ArrayType]]
