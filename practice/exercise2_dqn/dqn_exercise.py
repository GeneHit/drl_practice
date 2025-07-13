"""Enhanced DQN implementation with multi-environment support and 1D/2D observations.

This implementation supports:
- Multi-environment parallel training
- Both 1D (vector) and 2D (image) observations
- Experience replay buffer
- Target network for stability
- Configurable network architectures
"""

import copy
from typing import TypeAlias, Union, cast, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Discrete
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from practice.base import AgentBase, TrainerBase, ActType
from practice.exercise2_dqn.dqn_config import DQNConfig
from practice.utils.schedules import LinearSchedule

ObsType: TypeAlias = Union[np.uint8, np.float32]
ArrayType: TypeAlias = Union[np.bool_, np.float32]
EnvType: TypeAlias = gym.Env[NDArray[ObsType], ActType]
EnvsType: TypeAlias = gym.vector.VectorEnv[NDArray[ObsType], ActType, NDArray[ArrayType]]


class QNet1D(nn.Module):
    """Q network for 1D vector observations."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class QNet2D(nn.Module):
    """Q network with 2D convolution for image observations.
    
    Based on DeepMind's DQN architecture for Atari:
    https://www.nature.com/articles/nature14236
    """
    
    def __init__(self, in_shape: tuple[int, int, int], action_dim: int) -> None:
        super().__init__()
        c, h, w = in_shape
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calculate the output size of conv layers
        conv_out_size = self._get_conv_out_size(in_shape)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
    
    def _get_conv_out_size(self, shape: tuple[int, int, int]) -> int:
        """Calculate the output size of convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = self.conv(dummy_input)
            return int(np.prod(dummy_output.size()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate buffers
        self.states = torch.zeros((capacity, *obs_shape), device=device, dtype=torch.float32)
        self.actions = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.rewards = torch.zeros((capacity,), device=device, dtype=torch.float32)
        self.next_states = torch.zeros((capacity, *obs_shape), device=device, dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.bool, device=device)
    
    def add(self, states: NDArray[Any], actions: NDArray[Any], rewards: NDArray[Any], 
            next_states: NDArray[Any], dones: NDArray[Any]) -> None:
        """Add batch of experiences to buffer."""
        batch_size = len(states)
        
        # Handle wrap-around
        for i in range(batch_size):
            idx = (self.ptr + i) % self.capacity
            self.states[idx] = torch.from_numpy(states[i]).to(self.device, dtype=torch.float32)
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.next_states[idx] = torch.from_numpy(next_states[i]).to(self.device, dtype=torch.float32)
            self.dones[idx] = bool(dones[i])  # Convert to Python bool first
        
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample batch of experiences."""
        if self.size < batch_size:
            raise ValueError(f"Buffer size {self.size} < batch_size {batch_size}")
        
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
        self.q_network.eval()
    
    def action(self, state: NDArray[Any]) -> ActType:
        """Get greedy action for given state."""
        with torch.no_grad():
            if state.ndim == 1:  # 1D observation
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            else:  # 2D observation
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def save_model(self, pathname: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'device': str(self.device)
        }, pathname)
    
    @classmethod
    def load_from_checkpoint(cls, pathname: str, device: torch.device | None = None) -> "DQNAgent":
        """Load model from checkpoint."""
        if device is None:
            device = torch.device("cpu")
        
        checkpoint = torch.load(pathname, map_location=device)
        
        # For this implementation, we need to recreate the network
        # In a real implementation, we would save the network config with the model
        # For now, we'll attempt to infer from the saved model
        model_state = checkpoint['model_state_dict']
        
        # Try to infer network architecture from state dict
        first_layer_key = [k for k in model_state.keys() if 'network.0.weight' in k or 'conv.0.weight' in k][0]
        
        if 'conv' in first_layer_key:
            # This is a 2D network - we would need the input shape
            # For now, raise an error since we can't infer the input shape
            raise NotImplementedError("Loading 2D networks requires saving input shape configuration")
        else:
            # This is a 1D network
            first_layer = model_state['network.0.weight']
            state_dim = first_layer.shape[1]
            action_dim = model_state['network.' + str(len([k for k in model_state.keys() if 'network.' in k and '.weight' in k]) * 2 - 2) + '.weight'].shape[0]
            
            # Create network with inferred dimensions
            q_network = QNet1D(state_dim, action_dim)
            q_network.load_state_dict(model_state)
            q_network.to(device)
            
            return cls(q_network, device)


class DQNTrainer(TrainerBase[NDArray[Any]]):
    """DQN trainer with multi-environment support and experience replay."""
    
    def __init__(self, config: DQNConfig, q_network: nn.Module, device: torch.device) -> None:
        super().__init__(config, q_network, device)
        
        # Cast config to DQNConfig for type safety
        self.config = cast(DQNConfig, config)
        self.q_network = q_network
        
        # Initialize target network
        self.target_network = copy.deepcopy(q_network)
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(q_network.parameters(), lr=self.config.learning_rate)
        
        # Initialize epsilon schedule
        self.epsilon_schedule = LinearSchedule(
            min_value=self.config.end_epsilon,
            max_value=self.config.start_epsilon,
            duration=self.config.epsilon_decay_duration
        )
        
        # Initialize replay buffer (will be set up in train())
        self.replay_buffer: ReplayBuffer | None = None
        
        # Training statistics
        self.training_results = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
            "epsilons": [],
            "steps": []
        }
    
    def sync_target_net(self) -> None:
        """Synchronize target network with main network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def action(self, states: NDArray[Any], step: int, eval_mode: bool = False) -> NDArray[ActType]:
        """Get actions for batch of states with epsilon-greedy exploration."""
        if eval_mode:
            epsilon = 0.0
        else:
            epsilon = self.epsilon_schedule(step)
        
        batch_size = len(states)
        actions = np.zeros(batch_size, dtype=np.int64)
        
        # Determine number of actions from network
        if hasattr(self.q_network, 'fc'):
            # 2D network
            num_actions = self.q_network.fc[-1].out_features
        elif hasattr(self.q_network, 'network'):
            # 1D network
            num_actions = self.q_network.network[-1].out_features
        else:
            # Fallback - create temp env to get action space
            temp_env = gym.make(self.config.env_id, **self.config.env_kwargs)
            num_actions = temp_env.action_space.n
            temp_env.close()
        
        # Random actions with probability epsilon
        random_mask = np.random.rand(batch_size) < epsilon
        random_actions = np.random.randint(0, num_actions, size=batch_size)
        actions[random_mask] = random_actions[random_mask]
        
        # Greedy actions for the rest
        if not random_mask.all():
            greedy_mask = ~random_mask
            greedy_states = states[greedy_mask]
            
            with torch.no_grad():
                if greedy_states.size > 0:
                    state_tensor = torch.from_numpy(greedy_states).float().to(self.device)
                    q_values = self.q_network(state_tensor)
                    greedy_actions = q_values.argmax(dim=1).cpu().numpy()
                    actions[greedy_mask] = greedy_actions
        
        return actions
    
    def update(self) -> float:
        """Update Q-network using batch from replay buffer."""
        if self.replay_buffer is None or len(self.replay_buffer) < self.config.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * (~dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def create_environments(self) -> EnvsType:
        """Create vectorized environments."""
        if self.config.use_multi_processing:
            # Create vector environments using proper gymnasium syntax
            envs = gym.vector.AsyncVectorEnv([
                lambda: gym.make(self.config.env_id, **self.config.env_kwargs)
                for _ in range(self.config.num_envs)
            ])
        else:
            envs = gym.vector.SyncVectorEnv([
                lambda: gym.make(self.config.env_id, **self.config.env_kwargs)
                for _ in range(self.config.num_envs)
            ])
        
        return envs
    
    def train(self, log_dir: str) -> DQNAgent:
        """Train DQN agent with multi-environment support."""
        print(f"Training DQN with {self.config.num_envs} environments...")
        
        # Create environments
        envs = self.create_environments()
        
        # Initialize replay buffer
        obs_shape = envs.single_observation_space.shape
        self.replay_buffer = ReplayBuffer(
            self.config.replay_buffer_capacity,
            obs_shape,
            self.device
        )
        
        # Setup logging
        writer = SummaryWriter(log_dir)
        
        # Initialize environments
        states, _ = envs.reset()
        episode_rewards = np.zeros(self.config.num_envs)
        episode_lengths = np.zeros(self.config.num_envs)
        completed_episodes = 0
        
        # Training loop
        for step in tqdm(range(self.config.timesteps), desc="Training DQN"):
            # Get actions
            actions = self.action(states, step)
            
            # Environment step
            next_states, rewards, terminated, truncated, infos = envs.step(actions)
            dones = terminated | truncated
            
            # Store experiences
            self.replay_buffer.add(states, actions, rewards, next_states, dones)
            
            # Update episode statistics
            episode_rewards += rewards
            episode_lengths += 1
            
            # Handle completed episodes
            for i, done in enumerate(dones):
                if done:
                    self.training_results["episode_rewards"].append(float(episode_rewards[i]))
                    self.training_results["episode_lengths"].append(int(episode_lengths[i]))
                    
                    # Log to tensorboard every 100 episodes
                    if completed_episodes % 100 == 0:
                        writer.add_scalar("Episode/Reward", episode_rewards[i], completed_episodes)
                        writer.add_scalar("Episode/Length", episode_lengths[i], completed_episodes)
                    
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    completed_episodes += 1
            
            states = next_states
            
            # Training step
            if step >= self.config.update_start_step and step % self.config.train_interval == 0:
                loss = self.update()
                self.training_results["losses"].append(loss)
                
                if step % 1000 == 0:
                    writer.add_scalar("Training/Loss", loss, step)
                    epsilon = self.epsilon_schedule(step)
                    writer.add_scalar("Training/Epsilon", epsilon, step)
                    self.training_results["epsilons"].append(epsilon)
                    self.training_results["steps"].append(step)
            
            # Update target network
            if step % self.config.target_update_interval == 0:
                self.sync_target_net()
        
        envs.close()
        writer.close()
        
        # Save training results
        self.save_training_results()
        
        # Create and save agent
        agent = DQNAgent(self.q_network, self.device)
        if self.config.save_result:
            model_path = f"{self.config.output_dir}/{self.config.model_filename}"
            agent.save_model(model_path)
        
        return agent


def create_dqn_network(config: DQNConfig, device: torch.device) -> nn.Module:
    """Create appropriate DQN network based on configuration."""
    # Create temporary environment to inspect observation space
    temp_env = gym.make(config.env_id, **config.env_kwargs)
    obs_space = temp_env.observation_space
    action_space = temp_env.action_space
    temp_env.close()
    
    if not isinstance(action_space, Discrete):
        raise ValueError("DQN only supports discrete action spaces")
    
    action_dim = action_space.n
    
    if config.is_2d_observation() and len(obs_space.shape) >= 3:
        # 2D/Image observations
        if len(obs_space.shape) == 3:
            in_shape = obs_space.shape  # (C, H, W)
        else:
            in_shape = obs_space.shape[-3:]  # Take last 3 dimensions
        
        network = QNet2D(in_shape, action_dim)
    else:
        # 1D/Vector observations
        if len(obs_space.shape) == 1:
            state_dim = obs_space.shape[0]
        else:
            state_dim = int(np.prod(obs_space.shape))
        
        network = QNet1D(state_dim, action_dim, config.hidden_dim, config.num_layers)
    
    return network.to(device)


# Main function using enhanced CLI
if __name__ == "__main__":
    from practice.enhanced_cli import create_enhanced_main_function
    
    main = create_enhanced_main_function(
        algorithm_name="DQN",
        config_class=DQNConfig,
        trainer_class=DQNTrainer,
        model_loader=DQNAgent,
        network_factory=create_dqn_network,
        config_example="obs_1d_config.json"
    )
    
    main() 