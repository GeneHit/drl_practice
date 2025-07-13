"""Enhanced REINFORCE algorithm with advanced features."""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base import ActType, AgentBase, BaseConfig, TrainerBase
from practice.utils.schedules import ConstantSchedule, LinearSchedule


@dataclass
class EnhancedReinforceConfig(BaseConfig):
    """Configuration for Enhanced REINFORCE training."""

    # Enhanced REINFORCE specific parameters
    learning_rate: float
    use_baseline: bool = True
    baseline_decay: float = 0.99
    entropy_coef: float = 0.01

    # Beta scheduler for entropy
    use_beta_scheduler: bool = True
    initial_beta: float = 0.1
    final_beta: float = 0.001
    beta_decay_duration: int = 1000

    # Intrinsic rewards
    use_curiosity: bool = False
    curiosity_coef: float = 0.1
    use_reward_shaping: bool = False
    shaping_coef: float = 0.1

    # Network architecture
    hidden_dim: int = 128
    num_layers: int = 2

    def validate(self) -> None:
        """Validate Enhanced REINFORCE specific parameters."""
        if not (0 < self.learning_rate <= 1):
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")

        if not (0 <= self.baseline_decay <= 1):
            raise ValueError(f"baseline_decay must be in [0, 1], got {self.baseline_decay}")

        if self.entropy_coef < 0:
            raise ValueError(f"entropy_coef must be non-negative, got {self.entropy_coef}")

        if self.initial_beta < 0 or self.final_beta < 0:
            raise ValueError("Beta values must be non-negative")

        if self.curiosity_coef < 0:
            raise ValueError(f"curiosity_coef must be non-negative, got {self.curiosity_coef}")


class PolicyNetwork(nn.Module):
    """Policy network for Enhanced REINFORCE."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 128, num_layers: int = 2
    ) -> None:
        super().__init__()

        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return F.softmax(logits, dim=-1)


class CuriosityModule(nn.Module):
    """Simple curiosity module for intrinsic motivation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()

        self.action_dim = action_dim

        # Forward model: predict next state given current state and action
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # One-hot encode action
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        input_tensor = torch.cat([state, action_onehot], dim=-1)
        return self.forward_model(input_tensor)

    def compute_intrinsic_reward(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute intrinsic reward based on prediction error."""
        predicted_next_state = self.forward(state, action)
        prediction_error = F.mse_loss(predicted_next_state, next_state, reduction="none").mean(
            dim=-1
        )
        return prediction_error

    def update(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> float:
        """Update the forward model."""
        predicted_next_state = self.forward(state, action)
        loss = F.mse_loss(predicted_next_state, next_state)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class RewardShaping:
    """Simple reward shaping based on state differences."""

    def __init__(self, coef: float = 0.1) -> None:
        self.coef = coef
        self.prev_state: NDArray[Any] | None = None

    def compute_shaped_reward(self, state: NDArray[Any], reward: float) -> float:
        """Compute shaped reward based on state progression."""
        if self.prev_state is None:
            self.prev_state = state
            return reward

        # Simple shaping: reward movement in state space
        state_diff = np.linalg.norm(state - self.prev_state)
        shaped_reward = reward + self.coef * state_diff

        self.prev_state = state
        return shaped_reward

    def reset(self) -> None:
        """Reset for new episode."""
        self.prev_state = None


class EnhancedReinforceAgent(AgentBase[NDArray[Any]]):
    """Enhanced REINFORCE agent for evaluation."""

    def __init__(self, policy_network: nn.Module, device: torch.device) -> None:
        self.policy_network = policy_network
        self.device = device

    def action(self, state: NDArray[Any]) -> ActType:
        """Get action from policy network."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action_probs = self.policy_network(state_tensor)
            action_dist = Categorical(action_probs)
            return action_dist.sample().item()

    def save_model(self, pathname: str) -> None:
        """Save model to file."""
        torch.save(
            {"model_state_dict": self.policy_network.state_dict(), "device": self.device}, pathname
        )

    @classmethod
    def load_from_checkpoint(
        cls, pathname: str, device: torch.device | None = None
    ) -> "EnhancedReinforceAgent":
        """Load model from checkpoint."""
        checkpoint = torch.load(pathname, map_location=device)

        # Need to recreate network architecture - this is a limitation
        # In practice, you'd also save the architecture parameters
        raise NotImplementedError("Need to implement network architecture saving/loading")


class EpisodeBuffer:
    """Buffer for storing episode data with intrinsic rewards."""

    def __init__(self) -> None:
        self.states: list[NDArray[Any]] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.intrinsic_rewards: list[float] = []
        self.log_probs: list[torch.Tensor] = []
        self.entropies: list[torch.Tensor] = []

    def add(
        self,
        state: NDArray[Any],
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        intrinsic_reward: float = 0.0,
    ) -> None:
        """Add step data to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.intrinsic_rewards.append(intrinsic_reward)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)

    def get_returns(self, gamma: float, use_intrinsic: bool = False) -> list[float]:
        """Compute discounted returns."""
        returns = []
        G = 0.0

        rewards = (
            self.rewards
            if not use_intrinsic
            else [r + ir for r, ir in zip(self.rewards, self.intrinsic_rewards)]
        )

        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        return returns

    def clear(self) -> None:
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.intrinsic_rewards.clear()
        self.log_probs.clear()
        self.entropies.clear()

    def __len__(self) -> int:
        return len(self.states)


class EnhancedReinforceTrainer(TrainerBase[NDArray[Any]]):
    """Enhanced REINFORCE trainer with advanced features."""

    def __init__(self, state_dim: int, action_dim: int, device: torch.device) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.policy_network: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.episode_buffer = EpisodeBuffer()
        self.baseline_value = 0.0
        self.gamma = 0.0

        # Enhanced features
        self.curiosity_module: CuriosityModule | None = None
        self.reward_shaping: RewardShaping | None = None
        self.beta_schedule: LinearSchedule | ConstantSchedule | None = None

    def _init_training(self, config: EnhancedReinforceConfig) -> None:
        """Initialize training components."""
        # Create policy network
        self.policy_network = PolicyNetwork(
            self.state_dim, self.action_dim, config.hidden_dim, config.num_layers
        ).to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)

        # Initialize enhanced features
        if config.use_curiosity:
            self.curiosity_module = CuriosityModule(self.state_dim, self.action_dim).to(self.device)

        if config.use_reward_shaping:
            self.reward_shaping = RewardShaping(config.shaping_coef)

        if config.use_beta_scheduler:
            self.beta_schedule = LinearSchedule(
                config.final_beta, config.initial_beta, config.beta_decay_duration
            )
        else:
            self.beta_schedule = ConstantSchedule(config.entropy_coef)

        self.gamma = config.gamma
        self.baseline_value = 0.0

    def action(self, state: NDArray[Any], **kwargs: Any) -> ActType:
        """Get action for evaluation mode."""
        if self.policy_network is None:
            raise RuntimeError("Must call train() before using action()")

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state_tensor)
        action_dist = Categorical(action_probs)

        action = action_dist.sample()
        return action.item()

    def action_with_log_prob(
        self, state: NDArray[Any]
    ) -> tuple[ActType, torch.Tensor, torch.Tensor]:
        """Get action along with log probability and entropy for training."""
        if self.policy_network is None:
            raise RuntimeError("Must call train() before using action_with_log_prob()")

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state_tensor)
        action_dist = Categorical(action_probs)

        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), log_prob, entropy

    def compute_intrinsic_reward(
        self,
        state: NDArray[Any],
        action: int,
        next_state: NDArray[Any],
        config: EnhancedReinforceConfig,
    ) -> float:
        """Compute intrinsic reward from curiosity module."""
        if not config.use_curiosity or self.curiosity_module is None:
            return 0.0

        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action_tensor = torch.tensor([action], device=self.device)
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)

            intrinsic_reward = self.curiosity_module.compute_intrinsic_reward(
                state_tensor, action_tensor, next_state_tensor
            )

            return config.curiosity_coef * intrinsic_reward.item()

    def update_baseline(self, episode_return: float, config: EnhancedReinforceConfig) -> None:
        """Update baseline value using exponential moving average."""
        if config.use_baseline:
            self.baseline_value = (
                config.baseline_decay * self.baseline_value
                + (1 - config.baseline_decay) * episode_return
            )

    def update_policy(self, episode: int, config: EnhancedReinforceConfig) -> tuple[float, float]:
        """Update policy using Enhanced REINFORCE algorithm."""
        if len(self.episode_buffer) == 0:
            return 0.0, 0.0

        # Get current beta value
        current_beta = self.beta_schedule(episode)

        # Compute returns (including intrinsic rewards if enabled)
        use_intrinsic = config.use_curiosity or config.use_reward_shaping
        returns = self.episode_buffer.get_returns(self.gamma, use_intrinsic)
        returns_tensor = torch.tensor(returns, device=self.device)

        # Normalize returns
        if len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # Apply baseline
        if config.use_baseline:
            returns_tensor = returns_tensor - self.baseline_value

        # Compute policy loss
        log_probs = torch.stack(self.episode_buffer.log_probs)
        entropies = torch.stack(self.episode_buffer.entropies)

        policy_loss = -(log_probs * returns_tensor).mean()
        entropy_loss = -entropies.mean()

        total_loss = policy_loss + current_beta * entropy_loss

        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return policy_loss.item(), entropy_loss.item()

    def train(self, config: EnhancedReinforceConfig, log_dir: str) -> EnhancedReinforceAgent:
        """Train Enhanced REINFORCE agent."""
        self._init_training(config)

        # Create environment
        env = gym.make(config.env_id, **config.env_kwargs)

        # Setup logging
        writer = SummaryWriter(log_dir)

        # Training statistics
        episode_rewards = []
        intrinsic_rewards = []
        policy_losses = []
        entropy_losses = []

        for episode in tqdm(range(config.episodes), desc="Training Enhanced REINFORCE"):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_intrinsic_reward = 0.0
            episode_length = 0

            # Reset reward shaping for new episode
            if self.reward_shaping is not None:
                self.reward_shaping.reset()

            # Run episode
            for step in range(config.max_steps):
                action, log_prob, entropy = self.action_with_log_prob(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                # Apply reward shaping
                if self.reward_shaping is not None:
                    reward = self.reward_shaping.compute_shaped_reward(state, reward)

                # Compute intrinsic reward
                intrinsic_reward = self.compute_intrinsic_reward(state, action, next_state, config)

                # Store experience
                self.episode_buffer.add(state, action, reward, log_prob, entropy, intrinsic_reward)

                # Update curiosity module
                if self.curiosity_module is not None:
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    action_tensor = torch.tensor([action], device=self.device)
                    next_state_tensor = (
                        torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
                    )

                    curiosity_loss = self.curiosity_module.update(
                        state_tensor, action_tensor, next_state_tensor
                    )

                episode_reward += reward
                episode_intrinsic_reward += intrinsic_reward
                episode_length += 1
                state = next_state

                if terminated or truncated:
                    break

            # Update policy
            policy_loss, entropy_loss = self.update_policy(episode, config)

            # Update baseline
            total_return = (
                episode_reward + episode_intrinsic_reward
                if config.use_curiosity
                else episode_reward
            )
            self.update_baseline(total_return, config)

            # Clear episode buffer
            self.episode_buffer.clear()

            # Log statistics
            episode_rewards.append(episode_reward)
            intrinsic_rewards.append(episode_intrinsic_reward)
            policy_losses.append(policy_loss)
            entropy_losses.append(entropy_loss)

            # Log to tensorboard
            if episode % 100 == 0:
                writer.add_scalar("Episode/Reward", episode_reward, episode)
                writer.add_scalar("Episode/Intrinsic_Reward", episode_intrinsic_reward, episode)
                writer.add_scalar("Episode/Length", episode_length, episode)
                writer.add_scalar("Training/Policy_Loss", policy_loss, episode)
                writer.add_scalar("Training/Entropy_Loss", entropy_loss, episode)
                writer.add_scalar("Training/Beta", self.beta_schedule(episode), episode)

                if config.use_baseline:
                    writer.add_scalar("Training/Baseline", self.baseline_value, episode)

        env.close()
        writer.close()

        # Save training results
        if config.save_result:
            model_path = f"{config.output_dir}/{config.model_filename}"
            agent = EnhancedReinforceAgent(self.policy_network, self.device)
            agent.save_model(model_path)

        return EnhancedReinforceAgent(self.policy_network, self.device)
