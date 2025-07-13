"""Test Q-table configuration for quick testing."""

from practice.qtable import QTableConfig

config = QTableConfig(
    env_id="FrozenLake-v1",
    env_kwargs={"map_name": "4x4", "is_slippery": False, "render_mode": "rgb_array"},
    episodes=10,  # Just a few episodes for testing
    max_steps=99,
    gamma=0.95,
    eval_episodes=10,
    eval_seed=None,
    output_dir="results/test_qtable/",
    save_result=True,
    model_filename="qtable.pkl",
    repo_id="",
    learning_rate=0.7,
    min_epsilon=0.05,
    max_epsilon=1.0,
    decay_rate=0.0005,
)
