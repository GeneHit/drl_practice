"""Train the DQN agent.

Reference:
Algorithm: https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm
Code:https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
"""

import argparse
import copy
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from gymnasium.spaces import Discrete
from numpy.typing import NDArray

# from torch.utils.tensorboard import SummaryWriter
from common.base import PolicyBase
from common.evaluation_utils import evaluate_agent
from common.replay_buffer_utils import ReplayBuffer
from common.scheduler_utils import LinearSchedule
from hands_on.exercise2_dqn.config import DQNTrainConfig
from hands_on.utils.config_utils import load_config_from_json
from hands_on.utils.env_utils import ActType, make_1d_env, make_image_env
from hands_on.utils.numpy_tensor_utils import get_tensor_expanding_axis


class QNet2D(nn.Module):
    """Q network with 2D convolution."""

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

        # full connected layer
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(self.conv(x / 255.0))
        assert isinstance(y, torch.Tensor)  # make mypy happy
        return y


class QNet1D(nn.Module):
    """Q network with 1D discrete observation space."""

    def __init__(self, state_n: int, action_n: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_n, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_n),
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

    def action(self, state: Any, epsilon: float | None = None) -> ActType:
        if self._train_flag:
            assert epsilon is not None, "Epsilon is required in training mode"
            if random.random() < epsilon:
                # Exploration: take a random action with probability epsilon.
                return np.int32(random.randint(0, self._action_n - 1))

        # 2 case:
        # >> 1. need exploitation: take the action with the highest value.
        # >> 2. in the test phase, take the action with the highest value.
        assert isinstance(state, np.ndarray), "State must be a numpy array"
        state_tensor = get_tensor_expanding_axis(state).to(self._device)
        probs = self._q_network(state_tensor).cpu()
        return np.int32(probs.argmax().item())

    def get_score(self, state: Any, action: int | None = None) -> float:
        assert isinstance(state, np.ndarray), "State must be a numpy array"
        state_tensor = get_tensor_expanding_axis(state).to(self._device)
        probs = self._q_network(state_tensor).cpu()
        if action is None:
            return float(probs.max().item())
        return float(probs[0, action].item())

    def update(
        self, state: Any | None, action: Any | None, reward_target: Any
    ) -> None:
        assert isinstance(state, torch.Tensor), (
            "State must be a numpy array or torch tensor"
        )
        assert isinstance(action, torch.Tensor), (
            "Action must be a numpy array or torch tensor"
        )
        assert isinstance(reward_target, torch.Tensor), "Score must be a tensor"
        assert reward_target.dim() == 1, "Score must be a 1D tensor"

        state_tensor = state.to(self._device)
        actions = action.view(-1, 1).to(self._device)

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


def dqn_train_loop(
    env: gym.Env[NDArray[Any], ActType],
    dqn_agent: DQNAgent,
    device: torch.device,
    config: DQNTrainConfig,
) -> dict[str, Any]:
    """Train the DQN agent.

    Args:
        env (gym.Env): The environment.
        dqn_agent (DQNAgent): The DQN agent.
        device (torch.device): The device.
        train_config (DQNTrainConfig): The training configuration.
    """
    target_net = copy.deepcopy(dqn_agent.q_network).to(device)
    target_net.eval()
    epsilon_schedule = LinearSchedule(
        start_e=config.start_epsilon,
        end_e=config.end_epsilon,
        duration=int(config.exploration_fraction * config.global_steps),
    )
    replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)

    episode_reward: list[float] = []
    process_bar = tqdm.tqdm(range(config.global_steps))
    step = 0
    while step <= config.global_steps:
        epsilon = epsilon_schedule(step)
        state, _ = env.reset()
        rewards = 0.0
        for _ in range(config.max_steps):
            step += 1
            process_bar.update(1)
            action = dqn_agent.action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.add_one(
                state, action, float(reward), next_state, done
            )
            rewards += float(reward)
            if step >= config.update_start_step:
                if step % config.train_interval == 0:
                    experiences = replay_buffer.sample(config.batch_size)
                    states_batch = experiences.states.to(device)
                    reward_batch = experiences.rewards.to(device)
                    dones_batch = experiences.dones.to(device)
                    with torch.no_grad():
                        target_max, _ = target_net(states_batch).max(dim=1)
                        td_target = (
                            reward_batch.flatten()
                            + config.gamma
                            * target_max
                            * (1 - dones_batch.flatten().float())
                        )
                    dqn_agent.update(
                        state=experiences.states,
                        action=experiences.actions,
                        reward_target=td_target,
                    )
                    # clean the temp variables
                    del states_batch, reward_batch, dones_batch
                    del target_max, td_target, experiences
                if step % config.target_update_interval == 0:
                    # better: target = target * (1 - tau) + q_network * tau
                    target_net.load_state_dict(dqn_agent.q_network.state_dict())
            if done:
                break
            state = next_state

        episode_reward.append(rewards)
    process_bar.close()

    return {"episode_reward": episode_reward}


def dqn_train_main(
    env: gym.Env[NDArray[Any], ActType],
    cfg_data: dict[str, Any],
) -> None:
    # create the q_network, load the checkpoint if exists
    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    act_space = env.action_space
    assert isinstance(act_space, Discrete)  # make mypy happy
    action_n: int = int(act_space.n)
    if len(obs_shape) == 1:
        q_network: nn.Module = QNet1D(state_n=obs_shape[0], action_n=action_n)
    else:
        assert len(obs_shape) == 3, "The observation space must be 3D"
        q_network = QNet2D(in_shape=obs_shape, action_n=action_n)
    checkpoint_pathname = cfg_data.get("checkpoint_pathname", None)
    if checkpoint_pathname:
        q_network.load_state_dict(torch.load(checkpoint_pathname))

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    q_network = q_network.to(device)

    # create the dqn_agent
    lr = float(cfg_data["hyper_params"]["learning_rate"])
    dqn_agent = DQNAgent(
        q_network=q_network,
        optimizer=torch.optim.Adam(q_network.parameters(), lr=lr),
        action_n=action_n,
    )

    # train the dqn_agent
    start_time = time.time()
    train_result = dqn_train_loop(
        env=env,
        dqn_agent=dqn_agent,
        device=device,
        config=DQNTrainConfig.from_dict(cfg_data["hyper_params"]),
    )
    duration_min = (time.time() - start_time) / 60
    train_result["duration_min"] = duration_min

    # save the result
    if cfg_data["output_params"].get("save_result", False):
        # create the output directory
        output_params = cfg_data["output_params"]
        out_dir = Path(output_params["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        # save the model
        torch.save(
            dqn_agent.q_network.state_dict(),
            str(out_dir / output_params["model_filename"]),
        )
        # save the train result
        with open(out_dir / output_params["train_result_filename"], "w") as f:
            json.dump(train_result, f)
        # save all the config data
        cfg_data["env_params"].update({"device": str(device)})
        with open(out_dir / output_params["params_filename"], "w") as f:
            json.dump(cfg_data, f)

    # evaluate the agent
    mean_reward, std_reward = evaluate_agent(
        env=env,
        policy=dqn_agent,
        max_steps=int(cfg_data["hyper_params"]["max_steps"]),
        episodes=int(cfg_data["eval_params"]["eval_episodes"]),
        seed=tuple(cfg_data["eval_params"]["eval_seed"]),
    )
    # save the eval result
    if cfg_data["output_params"].get("save_result", False):
        eval_result = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(out_dir / output_params["eval_result_filename"], "w") as f:
            json.dump(eval_result, f)


def main(cfg_data: dict[str, Any]) -> None:
    # make the environment by the config
    env_params = cfg_data["env_params"]
    if env_params.get("use_image", False):
        env, more_env_params = make_image_env(
            env_id=env_params["env_id"],
            render_mode=env_params["render_mode"],
            resize_shape=tuple(env_params["resize_shape"]),
            frame_stack_size=env_params["frame_stack_size"],
        )
    else:
        env, more_env_params = make_1d_env(
            env_id=env_params["env_id"], render_mode=env_params["render_mode"]
        )
    # Update the original cfg_data with the new environment parameters
    cfg_data["env_params"].update(more_env_params)

    try:
        dqn_train_main(env=env, cfg_data=cfg_data)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_data = load_config_from_json(args.config)
    main(cfg_data=cfg_data)
