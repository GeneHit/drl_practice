"""Train the DQN agent.

Reference:
Algorithm: https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm
Code:https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Union, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Discrete
from numpy.typing import NDArray

# from torch.utils.tensorboard import SummaryWriter
from common.base import PolicyBase
from common.evaluation_utils import evaluate_agent
from common.numpy_tensor_utils import change_state_to_tensor
from common.replay_buffer_utils import ReplayBuffer
from common.scheduler_utils import LinearSchedule
from hands_on.exercise2_dqn.utils import describe_wrappers

EXERCISE2_RESULT_DIR = Path("results/exercise2_dqn/")

ActType = Union[np.integer[Any], int]


class QNet2D(nn.Module):
    """Q network with 2D convolution."""

    def __init__(self, in_shape: tuple[int, int, int], action_n: int) -> None:
        super().__init__()
        # in_c, h, w = in_shape
        # TODO: calculate n by h and w. Use 3136 for 84x84 now.
        n = 3136
        self.network = nn.Sequential(
            # [in_c, h, w] -> [32, h/4, w/4]
            nn.Conv2d(in_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n, 512),
            nn.ReLU(),
            nn.Linear(512, action_n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x / 255.0)
        assert isinstance(y, torch.Tensor)  # make mypy happy
        return y


class QNet1D(nn.Module):
    """Q network with 1D discrete observation space."""

    def __init__(self, state_n: int, action_n: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_n, 512),
            nn.ReLU(),
            nn.Linear(512, action_n),
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
        state_tensor = change_state_to_tensor(state).to(self._device)
        probs = self._q_network(state_tensor).cpu()
        return int(probs.argmax().item())

    def get_score(self, state: Any, action: int | None = None) -> float:
        assert isinstance(state, np.ndarray), "State must be a numpy array"
        state_tensor = change_state_to_tensor(state).to(self._device)
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

        state_tensor = change_state_to_tensor(state).to(self._device)
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

    # def _change_state_to_tensor(self, state: Any) -> torch.Tensor:
    #     if isinstance(state, np.ndarray):
    #         if state.ndim == 1:
    #             # Fix numpy array view call - use reshape instead of view
    #             state = state.reshape(1, -1)
    #         return torch.tensor(state, dtype=torch.float32)
    #     elif isinstance(state, torch.Tensor):
    #         return state
    #     else:
    #         raise ValueError("Invalid state type")


def dqn_train(
    env: gym.Env[NDArray[Any], ActType],
    dqn_agent: DQNAgent,
    global_steps: int,
    max_steps: int,
    device: torch.device,
    start_epsilon: float,
    end_epsilon: float,
    exploration_fraction: float,
    replay_buffer_capacity: int,
    batch_size: int,
    gamma: float,
    train_interval: int,
    target_update_interval: int,
    update_start_step: int,
) -> dict[str, Any]:
    """Train the DQN agent.

    Args:
        env (gym.Env): The environment.
        agent (DQNAgent): The DQN agent.
        global_step (int): The global step.
        max_steps (int): The maximum number of steps.
        device (torch.device): The device.
        start_epsilon (float): The start epsilon.
        end_epsilon (float): The end epsilon.
        exploration_fraction (float): The exploration fraction.
        replay_buffer_capacity (int): The replay buffer capacity.
        batch_size (int): The batch size.
        gamma (float): The gamma.
        train_interval (int): The train interval.
        target_update_interval (int): The target update interval.
        update_start_step (int): The update start step.

    Returns:
        dict[str, Any]: The metadata of the training.
    """
    import copy

    target_net = copy.deepcopy(dqn_agent.q_network).to(device)
    target_net.eval()
    epsilon_schedule = LinearSchedule(
        start_e=start_epsilon,
        end_e=end_epsilon,
        duration=int(exploration_fraction * global_steps),
    )
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    steps = 0
    episode_reward: list[float] = []
    while steps < global_steps:
        epsilon = epsilon_schedule(steps)
        state, _ = env.reset()
        rewards = 0.0
        for _ in range(max_steps):
            steps += 1
            action = dqn_agent.action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.add_one(
                state, action, float(reward), next_state, done
            )
            rewards += float(reward)
            if steps >= update_start_step:
                if steps % train_interval == 0:
                    experiences = replay_buffer.sample(batch_size)
                    with torch.no_grad():
                        target_max, _ = target_net(experiences.states).max(
                            dim=1
                        )
                        td_target = (
                            experiences.rewards.flatten()
                            + gamma
                            * target_max
                            * (1 - experiences.dones.flatten())
                        )
                    dqn_agent.update(
                        state=experiences.states,
                        action=experiences.actions,
                        reward_target=td_target,
                    )
                if steps % target_update_interval == 0:
                    # better: target = target * (1 - tau) + q_network * tau
                    target_net.load_state_dict(dqn_agent.q_network.state_dict())
            if done:
                break
            state = next_state

        episode_reward.append(rewards)

    return {"episode_reward": episode_reward}


def main(env: gym.Env[NDArray[Any], ActType], model_pathname: str) -> None:
    # Training parameters
    global_steps = 100000  # the total steps to train
    learning_rate = 1e-4  # Learning rate
    max_steps = 99  # Max steps per episode
    gamma = 0.99  # Discounting rate

    # Exploration parameters
    start_epsilon = 1.0  # Exploration probability at start
    end_epsilon = 0.05  # Minimum exploration probability
    # the fraction of `total-timesteps` it takes from start-e to go end-e
    exploration_fraction = 0.5
    replay_buffer_capacity = global_steps // 10
    batch_size = 32
    train_interval = 4
    target_update_interval = 1000
    update_start_step = 800

    act_shape = env.action_space.shape
    assert act_shape is not None
    act_space = env.action_space
    assert isinstance(act_space, Discrete)  # make mypy happy
    action_n: int = int(act_space.n)
    if len(act_shape) == 1:
        q_network: nn.Module = QNet1D(state_n=act_shape[0], action_n=action_n)
    else:
        assert len(act_shape) == 3, "The action space must be 3D"
        q_network = QNet2D(in_shape=act_shape, action_n=action_n)
    if model_pathname:
        q_network.load_state_dict(torch.load(model_pathname))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # use mps if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    q_network = q_network.to(device)
    dqn_agent = DQNAgent(
        q_network=q_network,
        optimizer=torch.optim.Adam(q_network.parameters(), lr=learning_rate),
        action_n=action_n,
    )

    train_result = dqn_train(
        env=env,
        dqn_agent=dqn_agent,
        global_steps=global_steps,
        max_steps=max_steps,
        device=device,
        start_epsilon=start_epsilon,
        end_epsilon=end_epsilon,
        exploration_fraction=exploration_fraction,
        replay_buffer_capacity=replay_buffer_capacity,
        batch_size=batch_size,
        gamma=gamma,
        train_interval=train_interval,
        target_update_interval=target_update_interval,
        update_start_step=update_start_step,
    )

    # save the model
    torch.save(
        dqn_agent.q_network.state_dict(),
        str(EXERCISE2_RESULT_DIR / "dqn_model.pth"),
    )

    # save the hyperparameters
    hyperparameters = {
        "env_id": "FrozenLake-v1",
        "global_training_steps": global_steps,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "gamma": gamma,
        "start_epsilon": start_epsilon,
        "end_epsilon": end_epsilon,
        "exploration_fraction": exploration_fraction,
        "replay_buffer_capacity": replay_buffer_capacity,
        "batch_size": batch_size,
        "train_interval": train_interval,
        "target_update_interval": target_update_interval,
        "update_start_step": update_start_step,
        "device": str(device),
    }
    with open(EXERCISE2_RESULT_DIR / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f)

    # Evaluation parameters
    eval_episodes = 100  # Total number of test episodes
    # The evaluation seed of the environment
    eval_seed: list[int] = list(
        random.randint(0, 1000000) for _ in range(eval_episodes)
    )

    mean_reward, std_reward = evaluate_agent(
        env=env,
        policy=dqn_agent,
        max_steps=max_steps,
        episodes=eval_episodes,
        seed=eval_seed,
    )
    # save the eval result
    eval_result = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "eval_episodes": eval_episodes,
        "eval_seed": eval_seed,
        "max_steps": max_steps,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    result = {
        "train_result": train_result,
        "eval_result": eval_result,
    }
    with open(EXERCISE2_RESULT_DIR / "result.json", "w") as f:
        json.dump(result, f)


def make_env() -> tuple[gym.Env[NDArray[Any], ActType], dict[str, Any]]:
    """Make the 2D environment.

    env.action_space.n=np.int64(4)
    env.observation_space.shape=(4, 84, 84)
    """
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.AddRenderObservation(env, render_only=True)
    env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
    # -> [**shape, 3] -> [**shape, 1]
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    # -> [**shape, 1] -> [4, **shape, 1]
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    transposed_space = gym.spaces.Box(
        low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
    )
    # -> [num_stack, **shape, 1] -> [num_stack, **shape]
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: obs.squeeze(-1),
        observation_space=transposed_space,
    )
    # env = TransformObservation(env, lambda obs: np.transpose(obs, (2, 0, 1)))
    env = cast(gym.Env[NDArray[Any], ActType], env)  # make mypy happy

    act_space = env.action_space
    assert isinstance(act_space, Discrete)  # make mypy happy
    env_params = {
        "env_id": "LunarLander-v3",
        "render_mode": "rgb_array",
        "wrappers": describe_wrappers(env),
        "observation_space.shape": env.observation_space.shape,
        "action_space": act_space.n,
        "observation_shape": (4, 84, 84),
    }
    return env, env_params


def make_1d_env() -> tuple[gym.Env[NDArray[Any], ActType], dict[str, Any]]:
    """Make the 1D environment.

    env.action_space.n=np.int64(4)
    env.observation_space.shape=(8,)
    """
    env = gym.make("LunarLander-v3")
    act_space = env.action_space
    assert isinstance(act_space, Discrete)  # make mypy happy
    env_params = {
        "env_id": "LunarLander-v3",
        "observation_space.shape": env.observation_space.shape,
        "action_space": act_space.n,
    }
    return env, env_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pathname", type=str, default="")
    parser.add_argument("--use_2d", action="store_true")
    args = parser.parse_args()

    env, env_params = make_env() if args.use_2d else make_1d_env()

    try:
        main(env=env, model_pathname=args.model_pathname)
        # save the env_params
        with open(EXERCISE2_RESULT_DIR / "env_params.json", "w") as f:
            json.dump(env_params, f)
    finally:
        env.close()
