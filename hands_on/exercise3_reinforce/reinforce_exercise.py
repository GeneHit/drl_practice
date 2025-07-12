import time
from typing import Any, Sequence, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hands_on.base import ActType
from hands_on.exercise2_dqn.dqn_exercise import EnvsType, ObsType
from hands_on.exercise3_reinforce.config import ReinforceConfig
from hands_on.utils.env_utils import extract_episode_data_from_infos


class Reinforce1DNet(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 128, layer_num: int = 2
    ) -> None:
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = self.network(x)  # make mypy happy
        return y


class ReinforceTrainer:
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gamma: float,
        state_shape: tuple[int, ...],
        grad_acc: int = 1,
    ) -> None:
        """Initialize the trainer.

        Args:
            net: The policy network.
            optimizer: The optimizer.
            device: The device to run the network.
            gamma: The discount factor.
            state_shape: The shape of the state.
            grad_acc: The number of gradient accumulation steps.
        """
        self._net = net
        self._optimizer = optimizer
        self._device = device
        self._gamma = gamma
        self._state_shape = state_shape
        self._grad_acc = grad_acc
        self._episode_count = 0

        # Track episodes since last optimizer step
        self._accumulated_episodes = 0
        # Initialize gradients
        self._optimizer.zero_grad()

    def action_and_log_prob(
        self, state: NDArray[ObsType], actions: Sequence[ActType] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Get action(s) and log_prob(s) for state(s).

        Args:
            state: Single state or batch of states
            actions: If provided, compute log probs for these actions. If None, sample new actions.
            eval: If True, use greedy policy (only used when actions=None)

        Returns:
            actions: Tensor
                If actions was None: newly sampled actions
                If actions was provided: the same actions as a tensor
            log_probs: Tensor
                Log probabilities for the actions
        """
        is_single = len(state.shape) == len(self._state_shape)
        state_batch = state if not is_single else state.reshape(1, *state.shape)
        state_tensor = torch.from_numpy(state_batch).to(self._device)

        self._net.train()  # Ensure network is in training mode
        probs = self._net(state_tensor)
        dist = torch.distributions.Categorical(probs)

        if actions is None:
            # Sample new actions
            sampled_actions = dist.sample()
            log_probs = dist.log_prob(sampled_actions)
            return sampled_actions, log_probs
        else:
            # Compute log probs for given actions
            action_tensor = torch.tensor(actions, device=self._device)
            log_probs = dist.log_prob(action_tensor)
            return action_tensor, log_probs

    def action(self, state: NDArray[ObsType]) -> NDArray[ActType]:
        """Sample actions.

        Args:
            state: Single state or batch of states

        Returns:
            actions: Sampled actions as numpy array
        """
        action_tensor = self.action_and_log_prob(state, actions=None)[0]
        actions_numpy = action_tensor.cpu().numpy().astype(ActType)
        return cast(NDArray[ActType], actions_numpy)

    def update(self, rewards: Sequence[float], log_probs: Tensor) -> float:
        """Update the policy network with a episode's data.

        Don't support inputting batch episodes now, because it's complex
        to collect the multi-episode batch data:
        1. for single env, use gradient accumulation is a better way.
        2. for multi env, different env has different episode length.
            - so it's complex to collect the multi-episode batch data.
            - if use the same episode length, it's not efficient.

        Gradient accumulation is useful for:
        1. Single env: accumulate multiple episodes in serial.
        2. Multi env: accumulate episodes from parallel environments.

        Parameters
        ----------
            rewards: Sequence[float]
                rewards of a episode, [episode_length, ]
            log_probs: Tensor
                log probabilities of actions in a episode, [episode_length, ]

        Returns
        -------
            loss: The policy loss value for logging
        """
        # Calculate returns (discounted cumulative rewards)
        returns: list[float] = []
        G = 0.0
        for reward in reversed(rewards):
            G = reward + self._gamma * G
            returns.insert(0, G)

        # Convert to tensor and normalize
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self._device)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # Calculate policy loss using vectorized operations
        policy_loss = -log_probs * returns_tensor
        total_loss = policy_loss.sum()

        # Scale loss by gradient accumulation factor to maintain proper gradient magnitude
        scaled_loss = total_loss / self._grad_acc

        # Accumulate gradients
        scaled_loss.backward()

        self._episode_count += 1
        self._accumulated_episodes += 1

        # Perform optimizer step when we've accumulated enough episodes
        if self._accumulated_episodes >= self._grad_acc:
            self._optimizer.step()
            self._optimizer.zero_grad()
            self._accumulated_episodes = 0

        return float(total_loss.item())

    def flush_gradients(self) -> None:
        """Force an optimizer step with accumulated gradients.

        Useful at the end of training to ensure all accumulated gradients are applied.
        """
        if self._accumulated_episodes > 0:
            self._optimizer.step()
            self._optimizer.zero_grad()
            self._accumulated_episodes = 0


class EpisodeBuffer:
    """Simple buffer to store episode data for REINFORCE."""

    def __init__(self) -> None:
        self.states: list[NDArray[Any]] = []
        self.actions: list[ActType] = []
        self.rewards: list[float] = []

    def add(self, state: NDArray[Any], action: ActType, reward: float) -> None:
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)

    def get_episode_data(
        self,
    ) -> tuple[list[NDArray[Any]], list[ActType], list[float]]:
        return self.states, self.actions, self.rewards

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def __len__(self) -> int:
        return len(self.states)


def reinforce_train_loop(
    envs: EnvsType, net: nn.Module, device: torch.device, config: ReinforceConfig, log_dir: str
) -> None:
    """Train the policy network with multiple environments.

    Args:
        envs: Vector environment for training
        net: The policy network to train
        device: Device to run computations on
        config: Training configuration
        log_dir: Directory for tensorboard logs.
    """
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)

    # Get environment info
    obs_shape = envs.single_observation_space.shape
    act_space = envs.single_action_space
    assert obs_shape is not None
    assert isinstance(act_space, gym.spaces.Discrete)

    # Create trainer
    trainer = ReinforceTrainer(
        net=net,
        optimizer=torch.optim.Adam(net.parameters(), lr=config.lr),
        device=device,
        gamma=config.gamma,
        state_shape=obs_shape,
        grad_acc=config.grad_acc,
    )

    # Initialize environments
    states, _ = envs.reset()
    assert isinstance(states, np.ndarray), "States must be numpy array"

    # Create variables for loop
    num_envs = len(states)
    env_episode_buffers = [EpisodeBuffer() for _ in range(num_envs)]
    pre_done: NDArray[np.bool_] = np.zeros(num_envs, dtype=np.bool_)
    pbar = tqdm(total=config.global_episode, desc="Training")
    episodes_completed = 0
    episode_reward_step = 0
    start_time = time.time()
    global_step = 0

    while episodes_completed < config.global_episode:
        # Sample actions for all environments
        actions = trainer.action(states)

        # Step environments
        next_states, rewards, terms, truncs, infos = envs.step(actions.tolist())
        dones = np.logical_or(terms, truncs)

        # Store data in episode buffers for environments that were not done in previous step
        # This avoids storing invalid reset state transitions
        for env_idx in range(num_envs):
            if not pre_done[env_idx]:  # Only store if environment wasn't done in previous step
                env_episode_buffers[env_idx].add(
                    states[env_idx], actions[env_idx].item(), float(rewards[env_idx])
                )

        # Updates for next iteration
        states = next_states
        pre_done = dones

        # Check for completed episodes and process them
        for env_idx in range(num_envs):
            env_buffer = env_episode_buffers[env_idx]
            if dones[env_idx] and len(env_buffer) > 0:
                # Process the completed episode if we have data
                # Get episode data from buffer
                episode_states, episode_actions, episode_rewards = env_buffer.get_episode_data()

                # Get log probs for the actual actions taken during episode
                # The target is get the seperated episode back-prop for each env
                episode_log_probs = trainer.action_and_log_prob(
                    state=np.array(episode_states), actions=episode_actions
                )[1]

                # Pass the tensor directly instead of converting to list
                loss = trainer.update(episode_rewards, episode_log_probs)

                # Log training metrics
                writer.add_scalar("losses/td_loss", loss, episodes_completed)
                episodes_completed += 1
                pbar.update(1)

                # Clear episode buffer for this environment
                env_buffer.clear()

        # Get episode statistics from RecordEpisodeStatistics wrapper
        # Use the new utility function to extract episode data
        ep_rewards, ep_lengths = extract_episode_data_from_infos(infos)
        # Log episode metrics when episodes complete
        for idx, reward in enumerate(ep_rewards):
            writer.add_scalar("episode/reward", reward, episode_reward_step)
            writer.add_scalar("episode/length", ep_lengths[idx], episode_reward_step)
            episode_reward_step += 1

        global_step += 1
        writer.add_scalar("charts/SPS", int(global_step / time.time() - start_time), global_step)

    # Flush any remaining accumulated gradients at the end of training
    trainer.flush_gradients()
    pbar.close()

    # Close writer
    writer.close()
