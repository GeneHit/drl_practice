"""2D observation configuration for DQN on LunarLander with image observations."""

from practice.exercise2_dqn.dqn_config import DQNConfig

config = DQNConfig(
    # Environment parameters
    env_id="LunarLander-v3",
    env_kwargs={
        "use_image": True,
        "render_mode": "rgb_array",
        "resize_shape": [84, 84],
        "frame_stack_size": 4
    },
    
    # Training parameters
    timesteps=20000,  # Reduced for 2D training
    learning_rate=1e-4,
    gamma=0.99,
    start_epsilon=1.0,
    end_epsilon=0.01,
    exploration_fraction=0.1,
    
    # DQN specific parameters
    replay_buffer_capacity=40000,  # Reduced for memory efficiency with images
    batch_size=64,
    train_interval=2,  # Less frequent updates for stability
    target_update_interval=1000,  # More frequent target updates
    update_start_step=1000,
    num_envs=6,
    use_multi_processing=True,
    
    # Evaluation parameters
    eval_episodes=100,
    eval_seed=[465, 131, 72, 782, 153, 233, 639, 746, 366, 473, 103, 569, 842, 412, 688, 946, 450, 722, 45, 606, 343, 194, 231, 415, 393, 781, 774, 318, 263, 154, 198, 549, 429, 889, 442, 54, 730, 910, 171, 520, 489, 145, 229, 526, 566, 725, 384, 895, 581, 138, 757, 8, 454, 58, 976, 595, 325, 347, 671, 399, 753, 517, 104, 133, 12, 391, 910, 301, 220, 226, 668, 999, 211, 644, 209, 734, 138, 308, 601, 779, 247, 692, 626, 916, 874, 808, 151, 92, 412, 62, 878, 29, 741, 903, 703, 21, 351, 545, 292, 103],
    
    # Output parameters
    output_dir="results/exercise2_dqn/lunar_2d/",
    save_result=True,
    model_filename="dqn.pth",
    params_filename="params.json",
    train_result_filename="train_result.json",
    eval_result_filename="eval_result.json",
    
    # Hub parameters
    repo_id="dqn-2d-LunarLander-v3"
) 