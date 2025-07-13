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

from hands_on.base import ActType, RewardBase
from hands_on.exercise2_dqn.dqn_exercise import EnvType, ObsType
from hands_on.exercise3_reinforce.config import ReinforceConfig


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
        baseline_decay: float = 0.99,
        use_baseline: bool = False,
        entropy_coef: float = 0.01,
    ) -> None:
        """Initialize the trainer.

        Args:
            net: The policy network.
            optimizer: The optimizer.
            device: The device to run the network.
            gamma: The discount factor.
            state_shape: The shape of the state.
            grad_acc: The number of gradient accumulation steps.
            baseline_decay: Decay factor for the moving average baseline.
            use_baseline: Whether to use a baseline for variance reduction.
            entropy_coef: Coefficient for entropy bonus term.
        """
        self._net = net
        self._optimizer = optimizer
        self._device = device
        self._gamma = gamma
        self._state_shape = state_shape
        self._grad_acc = grad_acc
        self._episode_count = 0
        self._net.train()  # Ensure network is in training mode

        # Track episodes since last optimizer step
        self._accumulated_episodes = 0
        # Initialize gradients
        self._optimizer.zero_grad()

        # Baseline for variance reduction
        self._baseline_decay = baseline_decay
        self._baseline_value = 0.0
        self._baseline_initialized = False
        self._use_baseline = use_baseline

        # Entropy bonus
        self._entropy_coef = entropy_coef

    def action_and_log_prob(
        self, state: NDArray[ObsType], actions: Sequence[ActType] | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Get action(s), log_prob(s), and entropy for state(s).

        Args:
            state: Single state or batch of states
            actions: If provided, compute log probs for these actions. If None, sample new actions.

        Returns:
            actions: Tensor
                If actions was None: newly sampled actions
                If actions was provided: the same actions as a tensor
            log_probs: Tensor
                Log probabilities for the actions
            entropies: Tensor
                Entropy values for the policy distribution
        """
        is_single = len(state.shape) == len(self._state_shape)
        state_batch = state if not is_single else state.reshape(1, *state.shape)
        state_tensor = torch.from_numpy(state_batch).to(self._device)

        probs = self._net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        entropies = dist.entropy()

        if actions is None:
            # Sample new actions
            sampled_actions = dist.sample()
            log_probs = dist.log_prob(sampled_actions)
            return sampled_actions, log_probs, entropies
        else:
            # Compute log probs for given actions
            action_tensor = torch.tensor(actions, device=self._device)
            log_probs = dist.log_prob(action_tensor)
            return action_tensor, log_probs, entropies

    def action(self, state: NDArray[ObsType]) -> NDArray[ActType]:
        """Sample actions.

        Args:
            state: Single state or batch of states

        Returns:
            actions: Sampled actions as numpy array
        """
        with torch.no_grad():
            action_tensor, _, _ = self.action_and_log_prob(state, actions=None)
            actions_numpy = action_tensor.cpu().numpy().astype(ActType)
        return cast(NDArray[ActType], actions_numpy)

    def update(
        self, rewards: Sequence[float], log_probs: Tensor, entropies: Tensor
    ) -> tuple[float, float]:
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
            entropies: Tensor
                entropy values for the policy distribution, [episode_length, ]

        Returns
        -------
            policy_loss: The policy loss value for logging
            entropy_loss: The entropy loss value for logging
        """
        # Calculate returns (discounted cumulative rewards)
        returns: list[float] = [0.0] * len(rewards)
        disc_return_t = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            disc_return_t = rewards[i] + self._gamma * disc_return_t
            returns[i] = disc_return_t

        # Convert to tensor
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self._device)

        # Update baseline with the average return of this episode
        episode_return = returns_tensor[0].item()  # First element is the total return
        if not self._use_baseline:
            advantages = returns_tensor
        else:
            if not self._baseline_initialized:
                self._baseline_value = episode_return
                self._baseline_initialized = True
            else:
                # Constant baseline using exponential moving average:
                #       b_t = decay * b_{t-1} + (1-decay) * G_t
                self._baseline_value = (
                    self._baseline_decay * self._baseline_value
                    + (1 - self._baseline_decay) * episode_return
                )

            # Apply baseline to reduce variance: A(s,a) = G_t - b (constant baseline)
            advantages = returns_tensor - self._baseline_value

        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate policy loss and entropy loss
        pg_loss = -(log_probs * advantages).sum()
        entropy_loss = -self._entropy_coef * entropies.sum()
        total_loss = pg_loss + entropy_loss

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

        return float(pg_loss.item()), float(entropy_loss.item())

    def flush_gradients(self) -> None:
        """Force an optimizer step with accumulated gradients.

        Useful at the end of training to ensure all accumulated gradients are applied.
        """
        if self._accumulated_episodes > 0:
            self._optimizer.step()
            self._optimizer.zero_grad()
            self._accumulated_episodes = 0

    def get_baseline_value(self) -> float:
        """Get the current baseline value for logging purposes."""
        return self._baseline_value


class EpisodeBuffer:
    """Simple buffer to store episode data for REINFORCE."""

    def __init__(self) -> None:
        self._states: list[NDArray[Any]] = []
        self._actions: list[ActType] = []
        self._rewards: list[float] = []
        self._intrinsic_rewards: list[float] = []
        self._rewarder_contributions: dict[str, list[float]] = {}

    def add(self, state: NDArray[Any], action: ActType, reward: float) -> None:
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._intrinsic_rewards.append(0.0)  # Initialize with 0
        # Initialize rewarder contributions for this step
        for rewarder_name in self._rewarder_contributions:
            self._rewarder_contributions[rewarder_name].append(0.0)

    def add_intrinsic_reward(self, intrinsic_reward: float, rewarder_name: str = "unknown") -> None:
        """Add intrinsic reward to the last step. Should be called after add()."""
        if len(self._intrinsic_rewards) == 0:
            raise RuntimeError("No intrinsic rewards to add, please call add() first.")

        self._intrinsic_rewards[-1] += intrinsic_reward
        # Track individual rewarder contribution
        if rewarder_name not in self._rewarder_contributions:
            # Initialize this rewarder's history with zeros for previous steps
            self._rewarder_contributions[rewarder_name] = [0.0] * len(self._intrinsic_rewards)
        # Ensure the contribution list is the same length as intrinsic_rewards
        while len(self._rewarder_contributions[rewarder_name]) < len(self._intrinsic_rewards):
            self._rewarder_contributions[rewarder_name].append(0.0)
        self._rewarder_contributions[rewarder_name][-1] += intrinsic_reward

    def get_episode_data(
        self,
    ) -> tuple[list[NDArray[Any]], list[ActType], list[float]]:
        return self._states, self._actions, self._rewards

    def clear(self, writer: SummaryWriter | None = None, episodes_completed: int = 0) -> None:
        """Clear the episode buffer and optionally log rewarder data.

        Args:
            writer: Optional SummaryWriter for logging rewarder data
            episodes_completed: Episode count for logging
        """
        # Log rewarder data before clearing if writer is provided
        if writer is not None:
            self._record_scalars(writer, episodes_completed)

        # Clear all data
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._intrinsic_rewards.clear()
        self._rewarder_contributions.clear()

    def __len__(self) -> int:
        return len(self._states)

    def _record_scalars(self, writer: SummaryWriter, episodes_completed: int) -> None:
        """Private method to record all rewarder-related scalars to tensorboard."""
        if len(self._rewarder_contributions) > 0:
            # Log intrinsic and total rewards
            writer.add_scalar(
                "episode/intrinsic_reward", sum(self._intrinsic_rewards), episodes_completed
            )
            writer.add_scalar("episode/total_reward", sum(self._rewards), episodes_completed)

            # Log individual rewarder contributions
            for rewarder_name in self._rewarder_contributions:
                writer.add_scalar(
                    f"episode/intrinsic_reward_{rewarder_name}",
                    sum(self._rewarder_contributions[rewarder_name]),
                    episodes_completed,
                )


def reinforce_train_loop(
    env: EnvType,
    net: nn.Module,
    device: torch.device,
    config: ReinforceConfig,
    log_dir: str,
    rewarders: Sequence[RewardBase] = [],
) -> None:
    """Train the policy network with a single environment.

    Note: the rewards is used for the exercise4_curiosity.

    Args:
        env: Single environment for training
        net: The policy network to train
        device: Device to run computations on
        config: Training configuration
        log_dir: Directory for tensorboard logs.
        rewarders: the rewarders to use for training.
    """
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)

    # Get environment info
    obs_shape = env.observation_space.shape
    act_space = env.action_space
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
        baseline_decay=config.baseline_decay,
        entropy_coef=config.entropy_coef,
    )

    # Initialize environment
    state, _ = env.reset()
    assert isinstance(state, np.ndarray), "State must be numpy array"

    # Create variables for loop
    episode_buffer = EpisodeBuffer()
    pbar = tqdm(total=config.global_episode, desc="Training")
    episodes_completed = 0
    start_time = time.time()
    global_step = 0

    while episodes_completed < config.global_episode:
        # Reset episode buffer for new episode
        episode_buffer.clear()
        # Reset environment
        state, _ = env.reset()
        done = False

        # Run one episode
        while not done:
            # Sample action
            action = trainer.action(state.reshape(1, -1))[0]

            # Step environment
            next_state, reward, term, trunc, info = env.step(action)
            done = term or trunc

            # Calculate intrinsic rewards
            new_reward = float(reward)
            intrinsic_rewards = []
            for rewarder in rewarders:
                intrinsic_reward = rewarder.get_reward(
                    state.reshape(1, -1), next_state.reshape(1, -1), episodes_completed
                )[0]
                intrinsic_rewards.append(intrinsic_reward)
                new_reward += float(intrinsic_reward)

            # Store data in episode buffer
            episode_buffer.add(state, action, new_reward)

            # Add intrinsic rewards from each rewarder
            for idx, rewarder in enumerate(rewarders):
                episode_buffer.add_intrinsic_reward(
                    intrinsic_reward=float(intrinsic_rewards[idx]),
                    rewarder_name=rewarder.__class__.__name__,
                )

            # Update state
            state = next_state
            global_step += 1

        # Process completed episode
        if len(episode_buffer) > 0:
            # Get episode data from buffer
            episode_states, episode_actions, episode_rewards = episode_buffer.get_episode_data()

            # Get log probs and entropies for the actual actions taken during episode
            _, episode_log_probs, episode_entropies = trainer.action_and_log_prob(
                state=np.array(episode_states), actions=episode_actions
            )

            # Update policy
            policy_loss, entropy_loss = trainer.update(
                episode_rewards, episode_log_probs, episode_entropies
            )

            # Log training metrics
            writer.add_scalar("losses/policy_loss", policy_loss, episodes_completed)
            writer.add_scalar("losses/entropy_loss", entropy_loss, episodes_completed)
            writer.add_scalar("losses/total_loss", policy_loss + entropy_loss, episodes_completed)
            writer.add_scalar("losses/baseline", trainer.get_baseline_value(), episodes_completed)

            # Clear episode buffer and log rewarder data if present
            episode_buffer.clear(writer=writer, episodes_completed=episodes_completed)

            episodes_completed += 1
            pbar.update(1)

            # Log episode statistics if available in info
            if "episode" in info:
                writer.add_scalar("episode/reward", info["episode"]["r"], episodes_completed)
                writer.add_scalar("episode/length", info["episode"]["l"], episodes_completed)

        # Log performance metrics periodically
        if global_step % 100 == 0:
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

    # Flush any remaining accumulated gradients at the end of training
    trainer.flush_gradients()

    # Close writer
    writer.close()
