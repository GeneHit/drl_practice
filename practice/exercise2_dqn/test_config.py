"""Test configuration for DQN with minimal training time."""

from practice.exercise2_dqn.dqn_config import DQNConfig

config = DQNConfig(
    # Environment parameters
    env_id="CartPole-v1",
    env_kwargs={},
    # Training parameters (minimal for testing)
    timesteps=1000,  # Very short for testing
    learning_rate=1e-4,
    gamma=0.99,
    start_epsilon=1.0,
    end_epsilon=0.01,
    exploration_fraction=0.1,
    # DQN specific parameters
    replay_buffer_capacity=5000,
    batch_size=32,
    train_interval=1,
    target_update_interval=100,
    update_start_step=100,
    num_envs=2,  # Reduced for testing
    use_multi_processing=False,  # Use sync for testing
    # Evaluation parameters
    eval_episodes=10,
    eval_seed=None,
    # Output parameters
    output_dir="results/test_dqn/",
    save_result=True,
    model_filename="dqn.pth",
    params_filename="params.json",
    train_result_filename="train_result.json",
    eval_result_filename="eval_result.json",
    # Hub parameters
    repo_id="test-dqn",
)
