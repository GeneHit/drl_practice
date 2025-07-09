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
from torch import Tensor

# from torch.utils.tensorboard import SummaryWriter
from hands_on.base import ActType, PolicyBase, ScheduleBase
from hands_on.exercise2_dqn.config import DQNTrainConfig
from hands_on.exercise2_dqn.dqn_train import QNet1D, QNet2D
from hands_on.utils.env_utils import make_1d_env, make_image_env
from hands_on.utils.evaluation_utils import evaluate_agent
from hands_on.utils.file_utils import (
    load_config_from_json,
    save_model_and_result,
)
from hands_on.utils_exercise.numpy_tensor_utils import get_tensor_expanding_axis
from hands_on.utils_exercise.replay_buffer_utils import Experience, ReplayBuffer
from hands_on.utils_exercise.scheduler_utils import LinearSchedule

ObsType: TypeAlias = Union[np.uint8, np.float32]
ArrayType: TypeAlias = Union[np.bool_, np.float32]
EnvType: TypeAlias = gym.Env[NDArray[ObsType], ActType]
EnvsType: TypeAlias = gym.vector.VectorEnv[
    NDArray[ObsType], ActType, NDArray[ArrayType]
]


class DQNAgent(PolicyBase):
    """DQN agent for evaluation/gameplay.

    This agent is focused on action selection using a trained Q-network.
    It does not handle training-specific operations.
    """

    def __init__(self, q_network: nn.Module) -> None:
        self._q_network = q_network
        self._device = next(q_network.parameters()).device

    def set_train_flag(self, train_flag: bool) -> None:
        """Set whether the agent is in training mode.

        TODO: delete this method
        """
        pass

    def action(
        self, state: NDArray[ObsType], epsilon: float | None = None
    ) -> ActType:
        """Get action for single state."""
        # take the action with the highest value.
        self._q_network.eval()
        state_tensor = get_tensor_expanding_axis(state).to(self._device)
        with torch.no_grad():
            probs = self._q_network(state_tensor).cpu()
        return ActType(probs.argmax().item())

    def get_score(
        self, state: NDArray[ObsType], action: ActType | None = None
    ) -> float:
        """Get Q-value score for a single state-action pair."""
        raise NotImplementedError("DQNAgent does not implement get_score")

    def update(self, state: Any, action: Any, reward_target: Any) -> None:
        """Not implemented for evaluation agent."""
        raise NotImplementedError("DQNAgent does not implement update")

    def only_save_model(self, pathname: str) -> None:
        """Save the DQN model."""
        assert pathname.endswith(".pth")
        torch.save(self._q_network, pathname)

    @classmethod
    def load_model(cls, pathname: str) -> "DQNAgent":
        """Load the DQN model."""
        assert pathname.endswith(".pth")
        q_network = torch.load(pathname)
        return cls(q_network=q_network)


class DQNTrainer:
    """Handles DQN training operations."""

    def __init__(
        self,
        q_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gamma: float,
        epsilon: ScheduleBase,
        state_shape: tuple[int, ...],
        action_n: int,
    ) -> None:
        self._q_network = q_network
        self._target_net = copy.deepcopy(q_network)
        self._target_net.eval()
        self._optimizer = optimizer
        self._device = device
        self._gamma = gamma
        self._epsilon = epsilon
        self._state_shape = state_shape
        self._action_n = action_n

    def sync_target_net(self) -> None:
        """Synchronize target network with current Q-network."""
        self._target_net.load_state_dict(self._q_network.state_dict())

    def action(
        self, state: NDArray[ObsType], step: int, eval: bool = False
    ) -> NDArray[ActType]:
        """Get action(s) for state(s).

        Args:
            state: Single state or batch of states
            step: Current step in the training process

        Returns:
            actions: NDArray[ActType]
                Single action or batch of actions depending on input shape.
                If the input is a single state, output a (1, ) array, making
                the output type consistent.
        """
        # Check if input is a single state or batch of states
        is_single = len(state.shape) == len(self._state_shape)
        state_batch = state if not is_single else state.reshape(1, *state.shape)

        if eval:
            # Test phase: always greedy
            state_tensor = torch.from_numpy(state_batch).to(self._device)
            with torch.no_grad():
                q_values = self._q_network(state_tensor).cpu()
                actions = q_values.argmax(dim=1).numpy().astype(ActType)
            return cast(NDArray[ActType], actions)

        # Training phase: epsilon-greedy
        batch_size = state_batch.shape[0]
        actions = np.zeros(batch_size, dtype=ActType)

        # Random mask for exploration
        epsilon = self._epsilon(step)
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
            self._q_network.train()
            with torch.no_grad():
                q_values = self._q_network(state_tensor).cpu()
                greedy_actions = q_values.argmax(dim=1).numpy().astype(ActType)
                actions[~random_mask] = greedy_actions

        return cast(NDArray[ActType], actions)

    def update(self, experiences: Experience) -> None:
        """Update Q-network using experiences.

        Args:
            experiences: Batch of experiences from replay buffer
        """
        # Move all inputs to device
        states = experiences.states.to(self._device)
        actions = experiences.actions.view(-1, 1).to(self._device)
        rewards = experiences.rewards.to(self._device)
        next_states = experiences.next_states.to(self._device)
        dones = experiences.dones.to(self._device)

        # Compute TD target using target network
        with torch.no_grad():
            target_max: Tensor = self._target_net(next_states).max(dim=1)[0]
            td_target = rewards.flatten() + self._gamma * target_max * (
                1 - dones.flatten().float()
            )

        # Get current Q-values for the actions taken
        self._q_network.train()
        current_q = self._q_network(states).gather(1, actions).squeeze()

        # Compute loss and update
        self._optimizer.zero_grad()
        loss = F.mse_loss(current_q, td_target)
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
    q_network: nn.Module,
    device: torch.device,
    config: DQNTrainConfig,
) -> dict[str, Any]:
    """Train the DQN agent with multiple environments."""
    # Get environment info
    obs_shape = envs.single_observation_space.shape
    act_space = envs.single_action_space
    assert obs_shape is not None
    assert isinstance(act_space, Discrete)
    action_n: int = int(act_space.n)

    epsilon_schedule = LinearSchedule(
        start_e=config.start_epsilon,
        end_e=config.end_epsilon,
        duration=int(config.exploration_fraction * config.timesteps),
    )

    # Create optimizer inside the function
    optimizer = torch.optim.Adam(
        q_network.parameters(), lr=config.learning_rate
    )

    # Create trainer inside the function
    trainer = DQNTrainer(
        q_network=q_network,
        optimizer=optimizer,
        device=device,
        gamma=config.gamma,
        epsilon=epsilon_schedule,
        state_shape=obs_shape,
        action_n=action_n,
    )

    # Get state shape from environment
    replay_buffer = ReplayBuffer(
        capacity=config.replay_buffer_capacity,
        state_shape=obs_shape,
        state_dtype=np.float32,
        action_dtype=np.int64,
    )

    num_envs = envs.num_envs
    episode_rewards: list[float] = []
    current_episode_rewards: NDArray[np.float32] = np.zeros(
        (num_envs,), dtype=np.float32
    )

    # Initialize environments
    states, _ = envs.reset()
    assert isinstance(states, np.ndarray), "States must be numpy array"

    for step in tqdm.tqdm(range(config.timesteps)):
        # Get actions for all environments using trainer's epsilon-greedy action method
        actions = trainer.action(states, step)
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

        # Add experiences to replay buffer in batch
        replay_buffer.add_batch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states.copy(),
            dones=dones,
        )

        # Handle episode completion
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            episode_rewards.extend(
                float(current_episode_rewards[i]) for i in done_indices
            )
            current_episode_rewards[done_indices] = 0.0

        # Training updates
        if step >= config.update_start_step:
            if step % config.train_interval == 0:
                experiences = replay_buffer.sample(config.batch_size)
                trainer.update(experiences)
            if step % config.target_update_interval == 0:
                trainer.sync_target_net()

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

    # Start training
    start_time = time.time()
    train_result = dqn_train_loop_multi_envs(
        envs=envs,
        q_network=q_network,
        device=device,
        config=DQNTrainConfig.from_dict(cfg_data["hyper_params"]),
    )
    envs.close()
    duration_min = (time.time() - start_time) / 60
    train_result["duration_min"] = duration_min

    # Evaluate the agent on a single environment
    dqn_agent = DQNAgent(q_network=q_network)
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

    # Save model and result if requested
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
