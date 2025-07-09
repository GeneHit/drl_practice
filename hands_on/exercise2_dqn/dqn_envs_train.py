"""Train the DQN agent with multiple environments.

Reference:
Algorithm: https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm
Code:https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
"""

import argparse
import copy
import time
from datetime import datetime
from typing import Any, Callable, TypeAlias, Union, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from gymnasium.spaces import Discrete
from numpy.typing import NDArray

# from torch.utils.tensorboard import SummaryWriter
from hands_on.base import ActType, PolicyBase
from hands_on.exercise2_dqn.config import DQNTrainConfig
from hands_on.exercise2_dqn.dqn_train import QNet1D, QNet2D
from hands_on.utils.env_utils import make_1d_env, make_image_env
from hands_on.utils.evaluation_utils import evaluate_agent
from hands_on.utils.file_utils import (
    load_config_from_json,
    save_model_and_result,
)
from hands_on.utils_exercise.replay_buffer_utils import ReplayBuffer
from hands_on.utils_exercise.scheduler_utils import LinearSchedule

ObsType: TypeAlias = Union[np.uint8, np.float32]
ArrayType: TypeAlias = Union[np.bool_, np.float32]
EnvType: TypeAlias = gym.Env[NDArray[ObsType], ActType]
EnvsType: TypeAlias = gym.vector.VectorEnv[
    NDArray[ObsType], ActType, NDArray[ArrayType]
]


class DQNAgent(PolicyBase):
    """DQN agent."""

    def __init__(
        self,
        q_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        state_shape: tuple[int, ...],
        action_n: int,
    ) -> None:
        self._q_network = q_network
        self._optimizer = optimizer
        self._train_flag = False
        self._device = next(q_network.parameters()).device
        self._action_n = action_n
        self._state_shape = state_shape

    @property
    def q_network(self) -> nn.Module:
        return self._q_network

    def set_train_flag(self, train_flag: bool) -> None:
        self._train_flag = train_flag
        self._q_network.train(train_flag)

    def only_save_model(self, pathname: str) -> None:
        """Save the DQN model."""
        # only save the q_network
        assert pathname.endswith(".pth")
        torch.save(self._q_network, pathname)

    def action(
        self, state: NDArray[ObsType], epsilon: float | None = None
    ) -> NDArray[ActType]:
        """Get action(s) for state(s).

        Args:
            state: Single state or batch of states
            epsilon: Exploration rate for epsilon-greedy policy

        Returns:
            actions: NDArray[ActType]
                Single action or batch of actions depending on input shape.
                If the input is a single state, output a (1, ) array, making
                the output type consistent.
        """
        # Check if input is a single state or batch of states
        is_single = len(state.shape) == len(self._state_shape)
        state_batch = state if not is_single else state.reshape(1, *state.shape)

        if not self._train_flag:
            # Test phase: always greedy
            state_tensor = torch.from_numpy(state_batch).to(self._device)
            with torch.no_grad():
                q_values = self._q_network(state_tensor).cpu()
                actions = q_values.argmax(dim=1).numpy().astype(ActType)
            return cast(NDArray[ActType], actions)

        # Training phase: epsilon-greedy
        assert epsilon is not None, "Epsilon is required in training mode"
        batch_size = state_batch.shape[0]
        actions = np.zeros(batch_size, dtype=ActType)

        # Random mask for exploration
        random_mask = np.random.random(batch_size) < epsilon

        # Random actions for exploration
        num_random = int(np.sum(random_mask))
        actions[random_mask] = np.random.randint(
            0, self._action_n, size=num_random, dtype=ActType
        )

        # Greedy actions for exploitation
        if not np.all(random_mask):
            exploit_states = state_batch[~random_mask]
            state_tensor = torch.from_numpy(exploit_states).to(self._device)
            with torch.no_grad():
                q_values = self._q_network(state_tensor).cpu()
                greedy_actions = q_values.argmax(dim=1).numpy().astype(ActType)
                actions[~random_mask] = greedy_actions

        return cast(NDArray[ActType], actions)

    def get_score(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get Q-value score(s) for state(s) and action(s).

        Args:
            state: Single state or batch of states as tensor
            action: Optional single action or batch of actions as tensor.
                    If None, returns max Q-value(s)

        Returns:
            Single score (float) or batch of scores (tensor)
        """
        # Check if input is a single state or batch of states
        is_single = len(state.shape) == len(self._state_shape)
        state_batch = state if not is_single else state.unsqueeze(0)
        assert action is None, "not implemented"

        # Move to device if needed
        state_batch = state_batch.to(self._device)
        # Get Q-values for all actions
        with torch.no_grad():
            scores = self._q_network(state_batch).max(dim=1)[0]

        return cast(torch.Tensor, scores)

    def update(
        self,
        state: torch.Tensor | None,
        action: torch.Tensor | None,
        reward_target: torch.Tensor,
    ) -> None:
        assert state is not None, "State must be provided"
        assert action is not None, "Action must be provided"
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


def make_vector_env(
    env_fn: Callable[[], EnvType],
    num_envs: int,
    use_multi_processing: bool = False,
) -> EnvsType:
    """Create a vector environment."""
    if use_multi_processing and num_envs > 1:
        return gym.vector.AsyncVectorEnv([env_fn for _ in range(num_envs)])

    return gym.vector.SyncVectorEnv([env_fn for _ in range(num_envs)])


def dqn_train_loop_multi_envs(
    envs: EnvsType,
    dqn_agent: DQNAgent,
    device: torch.device,
    config: DQNTrainConfig,
) -> dict[str, Any]:
    """Train the DQN agent with multiple environments."""
    target_net = copy.deepcopy(dqn_agent.q_network).to(device)
    target_net.eval()
    epsilon_schedule = LinearSchedule(
        start_e=config.start_epsilon,
        end_e=config.end_epsilon,
        duration=int(config.exploration_fraction * config.timesteps),
    )

    # Get state shape from environment
    obs_shape = envs.single_observation_space.shape
    assert obs_shape is not None
    replay_buffer = ReplayBuffer(
        capacity=config.replay_buffer_capacity, state_shape=obs_shape
    )

    num_envs = envs.num_envs
    episode_rewards: list[float] = []
    episode_counts = np.zeros(num_envs, dtype=np.int32)
    current_episode_rewards: NDArray[np.float32] = np.zeros(
        (num_envs,), dtype=np.float32
    )

    # Initialize environments
    states, _ = envs.reset()
    assert isinstance(states, np.ndarray), "States must be numpy array"
    dqn_agent.set_train_flag(True)

    for step in tqdm.tqdm(range(config.timesteps)):
        epsilon = epsilon_schedule(step)

        # Get actions for all environments
        actions = dqn_agent.action(states, epsilon)
        assert isinstance(actions, np.ndarray), (
            "Actions must be numpy array for batch"
        )

        # Step all environments
        next_states, rewards, terminated, truncated, _ = envs.step(
            actions.tolist()
        )

        dones = np.logical_or(terminated, truncated, dtype=np.bool_)
        # Update episode rewards
        current_episode_rewards = current_episode_rewards + rewards

        # Add experiences to replay buffer
        for i in range(num_envs):
            replay_buffer.add_one(
                states[i],
                int(actions[i]),
                float(rewards[i]),
                next_states[i],
                dones[i],
            )

            # Handle episode completion
            if dones[i]:
                episode_rewards.append(float(current_episode_rewards[i]))
                current_episode_rewards[i] = 0.0
                episode_counts[i] += 1

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

    return {"episode_rewards": episode_rewards}


def dqn_train_main_multi_envs(
    envs: EnvsType,
    env_fn: Callable[[], EnvType],
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
        state_shape=obs_shape,
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

    # Evaluate the agent on a single environment
    dqn_agent.set_train_flag(False)
    eval_env = env_fn()
    try:
        mean_reward, std_reward = evaluate_agent(
            env=eval_env,
            policy=dqn_agent,
            max_steps=cfg_data["eval_params"].get("max_steps", None),
            episodes=int(cfg_data["eval_params"]["eval_episodes"]),
            seed=tuple(cfg_data["eval_params"]["eval_seed"]),
        )
    finally:
        eval_env.close()

    # Save model and result
    save_result: bool = cfg_data["output_params"].get("save_result", False)
    if save_result:
        eval_result = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        cfg_data["env_params"].update({"device": str(device)})
        save_model_and_result(
            cfg_data, train_result, eval_result, agent=dqn_agent
        )


def main(cfg_data: dict[str, Any]) -> None:
    """Main function to setup environments and start training."""
    # Create environment factory function
    env_params = cfg_data["env_params"]
    use_image: bool = env_params.get("use_image", False)

    def env_fn() -> EnvType:
        if use_image:
            env, _ = make_image_env(
                env_id=env_params["env_id"],
                render_mode=env_params["render_mode"],
                resize_shape=tuple(env_params["resize_shape"]),
                frame_stack_size=env_params["frame_stack_size"],
            )
            return cast(EnvType, env)
        else:
            env, _ = make_1d_env(env_id=env_params["env_id"])
            return cast(EnvType, env)

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
        else make_1d_env(env_id=env_params["env_id"])
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
