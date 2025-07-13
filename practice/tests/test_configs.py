"""Test configuration classes."""

import pytest
import tempfile
from pathlib import Path

from practice.qtable import QTableConfig
from practice.dqn import DQNConfig
from practice.reinforce import ReinforceConfig
from practice.enhanced_reinforce import EnhancedReinforceConfig


class TestQTableConfig:
    """Test Q-table configuration."""
    
    def test_valid_config(self) -> None:
        """Test creating a valid Q-table config."""
        config = QTableConfig(
            env_id="FrozenLake-v1",
            env_kwargs={"map_name": "4x4", "is_slippery": False},
            episodes=1000,
            max_steps=99,
            gamma=0.95,
            eval_episodes=100,
            eval_seed=None,
            output_dir="results/test/",
            save_result=True,
            model_filename="qtable.pkl",
            repo_id="",
            learning_rate=0.7,
            min_epsilon=0.05,
            max_epsilon=1.0,
            decay_rate=0.0005
        )
        
        assert config.env_id == "FrozenLake-v1"
        assert config.episodes == 1000
        assert config.learning_rate == 0.7
        assert Path(config.output_dir).exists()
    
    def test_invalid_learning_rate(self) -> None:
        """Test invalid learning rate raises error."""
        with pytest.raises(ValueError, match="learning_rate must be in"):
            QTableConfig(
                env_id="FrozenLake-v1",
                env_kwargs={},
                episodes=1000,
                max_steps=99,
                gamma=0.95,
                eval_episodes=100,
                eval_seed=None,
                output_dir="results/test/",
                save_result=True,
                model_filename="qtable.pkl",
                repo_id="",
                learning_rate=1.5,  # Invalid
                min_epsilon=0.05,
                max_epsilon=1.0,
                decay_rate=0.0005
            )
    
    def test_invalid_epsilon_range(self) -> None:
        """Test invalid epsilon range raises error."""
        with pytest.raises(ValueError, match="min_epsilon must be <= max_epsilon"):
            QTableConfig(
                env_id="FrozenLake-v1",
                env_kwargs={},
                episodes=1000,
                max_steps=99,
                gamma=0.95,
                eval_episodes=100,
                eval_seed=None,
                output_dir="results/test/",
                save_result=True,
                model_filename="qtable.pkl",
                repo_id="",
                learning_rate=0.7,
                min_epsilon=0.9,  # Greater than max
                max_epsilon=0.1,
                decay_rate=0.0005
            )


class TestDQNConfig:
    """Test DQN configuration."""
    
    def test_valid_config(self) -> None:
        """Test creating a valid DQN config."""
        config = DQNConfig(
            env_id="CartPole-v1",
            env_kwargs={},
            episodes=500,
            max_steps=500,
            gamma=0.99,
            eval_episodes=100,
            eval_seed=None,
            output_dir="results/test/",
            save_result=True,
            model_filename="dqn.pth",
            repo_id="",
            learning_rate=0.001,
            min_epsilon=0.05,
            max_epsilon=1.0,
            epsilon_decay_duration=10000,
            target_network_frequency=1000,
            batch_size=32,
            buffer_size=10000,
            learning_starts=1000,
            train_frequency=4
        )
        
        assert config.env_id == "CartPole-v1"
        assert config.batch_size == 32
        assert config.buffer_size == 10000
    
    def test_invalid_batch_size(self) -> None:
        """Test invalid batch size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DQNConfig(
                env_id="CartPole-v1",
                env_kwargs={},
                episodes=500,
                max_steps=500,
                gamma=0.99,
                eval_episodes=100,
                eval_seed=None,
                output_dir="results/test/",
                save_result=True,
                model_filename="dqn.pth",
                repo_id="",
                learning_rate=0.001,
                min_epsilon=0.05,
                max_epsilon=1.0,
                epsilon_decay_duration=10000,
                target_network_frequency=1000,
                batch_size=-1,  # Invalid
                buffer_size=10000,
                learning_starts=1000,
                train_frequency=4
            )


class TestReinforceConfig:
    """Test REINFORCE configuration."""
    
    def test_valid_config(self) -> None:
        """Test creating a valid REINFORCE config."""
        config = ReinforceConfig(
            env_id="CartPole-v1",
            env_kwargs={},
            episodes=1000,
            max_steps=500,
            gamma=0.99,
            eval_episodes=100,
            eval_seed=None,
            output_dir="results/test/",
            save_result=True,
            model_filename="reinforce.pth",
            repo_id="",
            learning_rate=0.003,
            use_baseline=True,
            baseline_decay=0.99,
            entropy_coef=0.01
        )
        
        assert config.env_id == "CartPole-v1"
        assert config.use_baseline is True
        assert config.entropy_coef == 0.01
    
    def test_invalid_baseline_decay(self) -> None:
        """Test invalid baseline decay raises error."""
        with pytest.raises(ValueError, match="baseline_decay must be in"):
            ReinforceConfig(
                env_id="CartPole-v1",
                env_kwargs={},
                episodes=1000,
                max_steps=500,
                gamma=0.99,
                eval_episodes=100,
                eval_seed=None,
                output_dir="results/test/",
                save_result=True,
                model_filename="reinforce.pth",
                repo_id="",
                learning_rate=0.003,
                baseline_decay=1.5,  # Invalid
                entropy_coef=0.01
            )


class TestEnhancedReinforceConfig:
    """Test Enhanced REINFORCE configuration."""
    
    def test_valid_config(self) -> None:
        """Test creating a valid Enhanced REINFORCE config."""
        config = EnhancedReinforceConfig(
            env_id="MountainCar-v0",
            env_kwargs={},
            episodes=1000,
            max_steps=200,
            gamma=0.99,
            eval_episodes=100,
            eval_seed=None,
            output_dir="results/test/",
            save_result=True,
            model_filename="enhanced_reinforce.pth",
            repo_id="",
            learning_rate=0.003,
            use_baseline=True,
            use_curiosity=True,
            curiosity_coef=0.1,
            use_beta_scheduler=True,
            initial_beta=0.1,
            final_beta=0.001
        )
        
        assert config.env_id == "MountainCar-v0"
        assert config.use_curiosity is True
        assert config.curiosity_coef == 0.1
        assert config.use_beta_scheduler is True
    
    def test_invalid_beta_values(self) -> None:
        """Test invalid beta values raise error."""
        with pytest.raises(ValueError, match="Beta values must be non-negative"):
            EnhancedReinforceConfig(
                env_id="MountainCar-v0",
                env_kwargs={},
                episodes=1000,
                max_steps=200,
                gamma=0.99,
                eval_episodes=100,
                eval_seed=None,
                output_dir="results/test/",
                save_result=True,
                model_filename="enhanced_reinforce.pth",
                repo_id="",
                learning_rate=0.003,
                initial_beta=-0.1,  # Invalid
                final_beta=0.001
            ) 