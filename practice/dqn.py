"""Deep Q-Network (DQN) implementation."""

import copy
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base import AgentBase, BaseConfig, TrainerBase, ActType
from practice.utils.schedules import LinearSchedule


@dataclass
class DQNConfig(BaseConfig):
    """Configuration for DQN training."""
    
    # DQN specific parameters
    learning_rate: float
    min_epsilon: float
    max_epsilon: float
    epsilon_decay_duration: int
    target_network_frequency: int
    batch_size: int
    buffer_size: int
    learning_starts: int
    train_frequency: int
    
    # Network architecture
    hidden_dim: int = 128
    num_layers: int = 2
    
    def validate(self) -> None:
        """Validate DQN specific parameters."""
        if not (0 < self.learning_rate <= 1):
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        
        if not (0 <= self.min_epsilon <= 1):
            raise ValueError(f"min_epsilon must be in [0, 1], got {self.min_epsilon}")
        
        if not (0 <= self.max_epsilon <= 1):
            raise ValueError(f"max_epsilon must be in [0, 1], got {self.max_epsilon}")
        
        if self.min_epsilon > self.max_epsilon:
            raise ValueError("min_epsilon must be <= max_epsilon")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {self.buffer_size}")


class QNetwork(nn.Module):
    """Q-network for DQN."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate buffers
        self.states = torch.zeros((capacity, *obs_shape), device=device)
        self.actions = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.rewards = torch.zeros((capacity,), device=device)
        self.next_states = torch.zeros((capacity, *obs_shape), device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.bool, device=device)
    
    def add(self, state: NDArray[Any], action: int, reward: float, next_state: NDArray[Any], done: bool) -> None:
        """Add experience to buffer."""
        self.states[self.ptr] = torch.from_numpy(state).to(self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = torch.from_numpy(next_state).to(self.device)
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample batch of experiences."""
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )
    
    def __len__(self) -> int:
        return self.size


class DQNAgent(AgentBase[NDArray[Any]]):
    """DQN agent for evaluation."""
    
    def __init__(self, q_network: nn.Module, device: torch.device) -> None:
        self.q_network = q_network
        self.device = device
    
    def action(self, state: NDArray[Any]) -> ActType:
        """Get greedy action for given state."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def save_model(self, pathname: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'device': self.device
        }, pathname)
    
    @classmethod
    def load_from_checkpoint(cls, pathname: str, device: torch.device | None = None) -> "DQNAgent":
        """Load model from checkpoint."""
        checkpoint = torch.load(pathname, map_location=device)
        
        # Need to recreate network architecture - this is a limitation
        # In practice, you'd also save the architecture parameters
        raise NotImplementedError("Need to implement network architecture saving/loading")


class DQNTrainer(TrainerBase[NDArray[Any]]):
    """DQN trainer with experience replay and target network."""
    
    def __init__(self, state_dim: int, action_dim: int, device: torch.device) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.q_network: nn.Module | None = None
        self.target_network: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.replay_buffer: ReplayBuffer | None = None
        self.epsilon_schedule: LinearSchedule | None = None
        self.gamma = 0.0
    
    def _init_training(self, config: DQNConfig) -> None:
        """Initialize training components."""
        # Create networks
        self.q_network = QNetwork(
            self.state_dim, self.action_dim, 
            config.hidden_dim, config.num_layers
        ).to(self.device)
        
        self.target_network = QNetwork(
            self.state_dim, self.action_dim,
            config.hidden_dim, config.num_layers
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            config.buffer_size, (self.state_dim,), self.device
        )
        
        # Create epsilon schedule
        self.epsilon_schedule = LinearSchedule(
            config.min_epsilon, config.max_epsilon, config.epsilon_decay_duration
        )
        
        self.gamma = config.gamma
    
    def action(self, state: NDArray[Any], step: int = 0, eval_mode: bool = False) -> ActType:
        """Get action with epsilon-greedy exploration during training."""
        if self.q_network is None:
            raise RuntimeError("Must call train() before using action()")
        
        if eval_mode:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        
        epsilon = self.epsilon_schedule(step)
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def update(self, config: DQNConfig) -> float:
        """Update Q-network using batch from replay buffer."""
        if self.replay_buffer is None or len(self.replay_buffer) < config.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(config.batch_size)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, config: DQNConfig, log_dir: str) -> DQNAgent:
        """Train DQN agent."""
        self._init_training(config)
        
        # Create environment
        env = gym.make(config.env_id, **config.env_kwargs)
        
        # Setup logging
        writer = SummaryWriter(log_dir)
        
        # Training statistics
        episode_rewards = []
        losses = []
        
        state, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_count = 0
        
        for step in tqdm(range(config.episodes * config.max_steps), desc="Training DQN"):
            # Get action
            action = self.action(state, step)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Training step
            if step >= config.learning_starts and step % config.train_frequency == 0:
                loss = self.update(config)
                losses.append(loss)
                
                if step % 100 == 0:
                    writer.add_scalar("Training/Loss", loss, step)
            
            # Update target network
            if step % config.target_network_frequency == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Handle episode end
            if done or episode_length >= config.max_steps:
                episode_rewards.append(episode_reward)
                
                if episode_count % 100 == 0:
                    writer.add_scalar("Episode/Reward", episode_reward, episode_count)
                    writer.add_scalar("Episode/Length", episode_length, episode_count)
                    writer.add_scalar("Episode/Epsilon", self.epsilon_schedule(step), episode_count)
                
                # Reset for next episode
                state, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode_count += 1
                
                if episode_count >= config.episodes:
                    break
        
        env.close()
        writer.close()
        
        # Save training results
        if config.save_result:
            model_path = f"{config.output_dir}/{config.model_filename}"
            agent = DQNAgent(self.q_network, self.device)
            agent.save_model(model_path)
        
        return DQNAgent(self.q_network, self.device) 