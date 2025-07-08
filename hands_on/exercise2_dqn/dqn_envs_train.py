"""Train the DQN agent with multiple environments.

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
from typing import Any, Callable, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from gymnasium.spaces import Discrete
from gymnasium.vector import VectorEnv
from numpy.typing import NDArray

# from torch.utils.tensorboard import SummaryWriter
from hands_on.base import PolicyBase
from hands_on.exercise2_dqn.config import DQNTrainConfig
from hands_on.exercise2_dqn.dqn_train import QNet1D, QNet2D
from hands_on.utils.config_utils import load_config_from_json
from hands_on.utils.env_utils import make_1d_env, make_image_env
from hands_on.utils.evaluation_utils import evaluate_agent
from hands_on.utils_exercise.numpy_tensor_utils import get_tensor_expanding_axis
from hands_on.utils_exercise.replay_buffer_utils import ReplayBuffer
from hands_on.utils_exercise.scheduler_utils import LinearSchedule


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
        state_tensor = get_tensor_expanding_axis(state).to(self._device)
        probs = self._q_network(state_tensor).cpu()
        return int(probs.argmax().item())

    def action_batch(
        self, states: NDArray[Any], epsilon: float | None = None
    ) -> NDArray[np.int64]:
        """Get actions for a batch of states from multiple environments."""
        if self._train_flag:
            assert epsilon is not None, "Epsilon is required in training mode"
            # Handle epsilon-greedy for batch
            batch_size = states.shape[0]
            actions = np.zeros(batch_size, dtype=np.int64)

            # Random mask for exploration
            random_mask = np.random.random(batch_size) < epsilon

            # Random actions for exploration
            num_random = int(np.sum(random_mask))
            actions[random_mask] = np.random.randint(
                0, self._action_n, size=num_random, dtype=np.int64
            )

            # Greedy actions for exploitation
            if not np.all(random_mask):
                exploit_states = states[~random_mask]
                if len(exploit_states) > 0:
                    state_tensor = torch.from_numpy(exploit_states).to(
                        self._device
                    )
                    with torch.no_grad():
                        q_values = self._q_network(state_tensor).cpu()
                        greedy_actions = (
                            q_values.argmax(dim=1).numpy().astype(np.int64)
                        )
                        actions[~random_mask] = greedy_actions

            return actions
        else:
            # Test phase: always greedy
            state_tensor = torch.from_numpy(states).to(self._device)
            with torch.no_grad():
                q_values = self._q_network(state_tensor).cpu()
                greedy_actions = q_values.argmax(dim=1).numpy().astype(np.int64)
                return cast(NDArray[np.int64], greedy_actions)

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
        raise NotImplementedError("")


def make_vector_env(
    env_fn: Callable[[], gym.Env[NDArray[Any], np.integer[Any]]],
    num_envs: int,
    use_multi_processing: bool = False,
) -> VectorEnv[NDArray[Any], np.integer[Any], NDArray[Any]]:
    """Create a vector environment."""
    if use_multi_processing and num_envs > 1:
        return gym.vector.AsyncVectorEnv([env_fn for _ in range(num_envs)])

    return gym.vector.SyncVectorEnv([env_fn for _ in range(num_envs)])


def dqn_train_loop_multi_envs(
    envs: VectorEnv[NDArray[Any], np.integer[Any], NDArray[Any]],
    dqn_agent: DQNAgent,
    device: torch.device,
    config: DQNTrainConfig,
) -> dict[str, Any]:
    """Train the DQN agent with multiple environments.

    Args:
        envs (gym.vector.VectorEnv): The vector environment.
        dqn_agent (DQNAgent): The DQN agent.
        device (torch.device): The device.
        config (DQNTrainConfig): The training configuration.
    """
    target_net = copy.deepcopy(dqn_agent.q_network).to(device)
    target_net.eval()
    epsilon_schedule = LinearSchedule(
        start_e=config.start_epsilon,
        end_e=config.end_epsilon,
        duration=int(config.exploration_fraction * config.global_steps),
    )
    replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)

    num_envs = envs.num_envs
    episode_rewards: list[float] = []
    episode_counts = np.zeros(num_envs, dtype=np.int32)
    current_episode_rewards = np.zeros(num_envs, dtype=np.float32)

    process_bar = tqdm.tqdm(range(config.global_steps))
    step = 0

    # Initialize environments
    states, _ = envs.reset()
    assert isinstance(states, np.ndarray), "States must be numpy array"

    dqn_agent.set_train_flag(True)

    while step <= config.global_steps:
        epsilon = epsilon_schedule(step)

        # Get actions for all environments
        actions = dqn_agent.action_batch(states, epsilon)

        # Step all environments
        next_states, rewards, terminated, truncated, _ = envs.step(
            actions.tolist()
        )
        assert isinstance(next_states, np.ndarray), (
            "Next states must be numpy array"
        )
        assert isinstance(rewards, np.ndarray), "Rewards must be numpy array"
        assert isinstance(terminated, np.ndarray), (
            "Terminated must be numpy array"
        )
        assert isinstance(truncated, np.ndarray), (
            "Truncated must be numpy array"
        )

        dones = terminated | truncated

        # Update episode rewards
        current_episode_rewards += rewards

        # Add experiences to replay buffer
        for i in range(num_envs):
            replay_buffer.add_one(
                states[i],
                actions[i],
                float(rewards[i]),
                next_states[i],
                dones[i],
            )

            # Handle episode completion
            if dones[i]:
                episode_rewards.append(float(current_episode_rewards[i]))
                current_episode_rewards[i] = 0.0
                episode_counts[i] += 1

        step += num_envs  # Each step processes num_envs environments
        process_bar.update(num_envs)

        # Training updates
        if step >= config.update_start_step:
            if step % config.train_interval == 0:
                experiences = replay_buffer.sample(config.batch_size)
                states_batch = experiences.states.to(device)
                next_states_batch = experiences.next_states.to(device)
                reward_batch = experiences.rewards.to(device)
                dones_batch = experiences.dones.to(device)

                with torch.no_grad():
                    target_max, _ = target_net(next_states_batch).max(dim=1)
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

                # Clean temp variables
                del states_batch, next_states_batch, reward_batch, dones_batch
                del target_max, td_target, experiences

            if step % config.target_update_interval == 0:
                # Update target network
                target_net.load_state_dict(dqn_agent.q_network.state_dict())

        states = next_states

    process_bar.close()
    return {"episode_reward": episode_rewards}


def dqn_train_main_multi_envs(
    envs: gym.vector.VectorEnv[NDArray[Any], np.integer[Any], NDArray[Any]],
    env_fn: Callable[[], gym.Env[NDArray[Any], np.integer[Any]]],
    cfg_data: dict[str, Any],
) -> None:
    """Main training function for multi-environment DQN."""
    # Get environment info from a single environment
    obs_shape = envs.single_observation_space.shape
    act_space = envs.single_action_space
    assert obs_shape is not None  # make mypy happy
    assert isinstance(act_space, Discrete)  # make mypy happy
    action_n: int = int(act_space.n)

    # Create Q-network
    if len(obs_shape) == 1:
        q_network: nn.Module = QNet1D(state_n=obs_shape[0], action_n=action_n)
    else:
        assert len(obs_shape) == 3, "The observation space must be 3D"
        q_network = QNet2D(in_shape=obs_shape, action_n=action_n)

    # Load checkpoint if exists
    checkpoint_pathname = cfg_data.get("checkpoint_pathname", None)
    if checkpoint_pathname:
        q_network.load_state_dict(torch.load(checkpoint_pathname))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    q_network = q_network.to(device)

    # Create DQN agent
    lr = float(cfg_data["hyper_params"]["learning_rate"])
    dqn_agent = DQNAgent(
        q_network=q_network,
        optimizer=torch.optim.Adam(q_network.parameters(), lr=lr),
        action_n=action_n,
    )

    # Train the agent
    start_time = time.time()
    train_result = dqn_train_loop_multi_envs(
        envs=envs,
        dqn_agent=dqn_agent,
        device=device,
        config=DQNTrainConfig.from_dict(cfg_data["hyper_params"]),
    )
    envs.close()
    duration_min = (time.time() - start_time) / 60
    train_result["duration_min"] = duration_min

    # Save results
    save_result: bool = cfg_data["output_params"].get("save_result", False)
    if save_result:
        # Create output directory
        output_params = cfg_data["output_params"]
        out_dir = Path(output_params["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(
            dqn_agent.q_network.state_dict(),
            str(out_dir / output_params["model_filename"]),
        )

        # Save train result
        with open(out_dir / output_params["train_result_filename"], "w") as f:
            json.dump(train_result, f)

        # Save config
        cfg_data["env_params"].update({"device": str(device)})
        with open(out_dir / output_params["params_filename"], "w") as f:
            json.dump(cfg_data, f)

    # Evaluate the agent on a single environment
    dqn_agent.set_train_flag(False)
    eval_env = env_fn()
    try:
        mean_reward, std_reward = evaluate_agent(
            env=eval_env,
            policy=dqn_agent,
            max_steps=int(cfg_data["hyper_params"]["max_steps"]),
            episodes=int(cfg_data["eval_params"]["eval_episodes"]),
            seed=tuple(cfg_data["eval_params"]["eval_seed"]),
        )
    finally:
        eval_env.close()

    # Save evaluation result
    if save_result:
        eval_result = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(out_dir / output_params["eval_result_filename"], "w") as f:
            json.dump(eval_result, f)


def main(cfg_data: dict[str, Any]) -> None:
    """Main function to setup environments and start training."""
    # Create environment factory function
    env_params = cfg_data["env_params"]
    use_image: bool = env_params.get("use_image", False)

    def env_fn() -> gym.Env[NDArray[Any], np.integer[Any]]:
        if use_image:
            env, _ = make_image_env(
                env_id=env_params["env_id"],
                render_mode=env_params["render_mode"],
                resize_shape=tuple(env_params["resize_shape"]),
                frame_stack_size=env_params["frame_stack_size"],
            )
        else:
            env, _ = make_1d_env(
                env_id=env_params["env_id"],
                render_mode=env_params["render_mode"],
            )
        return env

    # Create vector environment
    num_envs: int = cfg_data["hyper_params"]["num_envs"]
    use_multi_processing: bool = cfg_data["hyper_params"][
        "use_multi_processing"
    ]
    envs = make_vector_env(env_fn, num_envs, use_multi_processing)

    # Get environment info and update config
    temp_env, more_env_info = (
        make_image_env(
            env_id=env_params["env_id"],
            render_mode=env_params["render_mode"],
            resize_shape=tuple(env_params["resize_shape"]),
            frame_stack_size=env_params["frame_stack_size"],
        )
        if use_image
        else make_1d_env(
            env_id=env_params["env_id"], render_mode=env_params["render_mode"]
        )
    )
    temp_env.close()
    cfg_data["env_params"].update(more_env_info)

    # Start training
    try:
        dqn_train_main_multi_envs(envs=envs, env_fn=env_fn, cfg_data=cfg_data)
    finally:
        if not envs.closed:
            envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_data = load_config_from_json(args.config)
    main(cfg_data=cfg_data)
