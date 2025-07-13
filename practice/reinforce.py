"""REINFORCE algorithm implementation."""

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base import AgentBase, BaseConfig, TrainerBase, ActType


@dataclass
class ReinforceConfig(BaseConfig):
    """Configuration for REINFORCE training."""
    
    # REINFORCE specific parameters
    learning_rate: float
    use_baseline: bool = False
    baseline_decay: float = 0.99
    entropy_coef: float = 0.01
    grad_accumulation_steps: int = 1
    
    # Network architecture
    hidden_dim: int = 128
    num_layers: int = 2
    
    def validate(self) -> None:
        """Validate REINFORCE specific parameters."""
        if not (0 < self.learning_rate <= 1):
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        
        if not (0 <= self.baseline_decay <= 1):
            raise ValueError(f"baseline_decay must be in [0, 1], got {self.baseline_decay}")
        
        if self.entropy_coef < 0:
            raise ValueError(f"entropy_coef must be non-negative, got {self.entropy_coef}")
        
        if self.grad_accumulation_steps <= 0:
            raise ValueError(f"grad_accumulation_steps must be positive, got {self.grad_accumulation_steps}")


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return F.softmax(logits, dim=-1)


class ReinforceAgent(AgentBase[NDArray[Any]]):
    """REINFORCE agent for evaluation."""
    
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
        torch.save({
            'model_state_dict': self.policy_network.state_dict(),
            'device': self.device
        }, pathname)
    
    @classmethod
    def load_from_checkpoint(cls, pathname: str, device: torch.device | None = None) -> "ReinforceAgent":
        """Load model from checkpoint."""
        checkpoint = torch.load(pathname, map_location=device)
        
        # Need to recreate network architecture - this is a limitation
        # In practice, you'd also save the architecture parameters
        raise NotImplementedError("Need to implement network architecture saving/loading")


class EpisodeBuffer:
    """Buffer for storing episode data."""
    
    def __init__(self) -> None:
        self.states: list[NDArray[Any]] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.log_probs: list[torch.Tensor] = []
        self.entropies: list[torch.Tensor] = []
    
    def add(self, state: NDArray[Any], action: int, reward: float, log_prob: torch.Tensor, entropy: torch.Tensor) -> None:
        """Add step data to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
    
    def get_returns(self, gamma: float) -> list[float]:
        """Compute discounted returns."""
        returns = []
        G = 0.0
        
        for reward in reversed(self.rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        return returns
    
    def clear(self) -> None:
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.entropies.clear()
    
    def __len__(self) -> int:
        return len(self.states)


class ReinforceTrainer(TrainerBase[NDArray[Any]]):
    """REINFORCE trainer with optional baseline."""
    
    def __init__(self, state_dim: int, action_dim: int, device: torch.device) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.policy_network: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.episode_buffer = EpisodeBuffer()
        self.baseline_value = 0.0
        self.gamma = 0.0
    
    def _init_training(self, config: ReinforceConfig) -> None:
        """Initialize training components."""
        # Create policy network
        self.policy_network = PolicyNetwork(
            self.state_dim, self.action_dim,
            config.hidden_dim, config.num_layers
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        
        self.gamma = config.gamma
        self.baseline_value = 0.0
    
    def action(self, state: NDArray[Any], **kwargs: Any) -> ActType:
        """Get action and store computation graph for training."""
        if self.policy_network is None:
            raise RuntimeError("Must call train() before using action()")
        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state_tensor)
        action_dist = Categorical(action_probs)
        
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return action.item()
    
    def action_with_log_prob(self, state: NDArray[Any]) -> tuple[ActType, torch.Tensor, torch.Tensor]:
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
    
    def update_baseline(self, episode_return: float, config: ReinforceConfig) -> None:
        """Update baseline value using exponential moving average."""
        if config.use_baseline:
            self.baseline_value = (config.baseline_decay * self.baseline_value + 
                                 (1 - config.baseline_decay) * episode_return)
    
    def update_policy(self, config: ReinforceConfig) -> tuple[float, float]:
        """Update policy using REINFORCE algorithm."""
        if len(self.episode_buffer) == 0:
            return 0.0, 0.0
        
        # Compute returns
        returns = self.episode_buffer.get_returns(self.gamma)
        returns_tensor = torch.tensor(returns, device=self.device)
        
        # Normalize returns (optional)
        if len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Apply baseline
        if config.use_baseline:
            returns_tensor = returns_tensor - self.baseline_value
        
        # Compute policy loss
        log_probs = torch.stack(self.episode_buffer.log_probs)
        entropies = torch.stack(self.episode_buffer.entropies)
        
        policy_loss = -(log_probs * returns_tensor).mean()
        entropy_loss = -entropies.mean()
        
        total_loss = policy_loss + config.entropy_coef * entropy_loss
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item(), entropy_loss.item()
    
    def train(self, config: ReinforceConfig, log_dir: str) -> ReinforceAgent:
        """Train REINFORCE agent."""
        self._init_training(config)
        
        # Create environment
        env = gym.make(config.env_id, **config.env_kwargs)
        
        # Setup logging
        writer = SummaryWriter(log_dir)
        
        # Training statistics
        episode_rewards = []
        policy_losses = []
        entropy_losses = []
        
        for episode in tqdm(range(config.episodes), desc="Training REINFORCE"):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            # Run episode
            for step in range(config.max_steps):
                action, log_prob, entropy = self.action_with_log_prob(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Store experience
                self.episode_buffer.add(state, action, reward, log_prob, entropy)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if terminated or truncated:
                    break
            
            # Update policy
            policy_loss, entropy_loss = self.update_policy(config)
            
            # Update baseline
            self.update_baseline(episode_reward, config)
            
            # Clear episode buffer
            self.episode_buffer.clear()
            
            # Log statistics
            episode_rewards.append(episode_reward)
            policy_losses.append(policy_loss)
            entropy_losses.append(entropy_loss)
            
            # Log to tensorboard
            if episode % 100 == 0:
                writer.add_scalar("Episode/Reward", episode_reward, episode)
                writer.add_scalar("Episode/Length", episode_length, episode)
                writer.add_scalar("Training/Policy_Loss", policy_loss, episode)
                writer.add_scalar("Training/Entropy_Loss", entropy_loss, episode)
                
                if config.use_baseline:
                    writer.add_scalar("Training/Baseline", self.baseline_value, episode)
        
        env.close()
        writer.close()
        
        # Save training results
        if config.save_result:
            model_path = f"{config.output_dir}/{config.model_filename}"
            agent = ReinforceAgent(self.policy_network, self.device)
            agent.save_model(model_path)
        
        return ReinforceAgent(self.policy_network, self.device) 