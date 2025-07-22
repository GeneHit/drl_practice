from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.env_typing import EnvType
from practice.exercise5_a2c.a2c_gae_exercise import A2CTrainer
from practice.exercise6_a3c.a3c_exercise import A3CConfig
from practice.utils.env_utils import get_device, get_env_from_config
from practice.utils_for_coding.agent_utils import A2CAgent
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> A3CConfig:
    """Get the application config."""
    # rollout_len = 32, so 150000 / 32 / 6 = 781
    global_step = 150000
    num_workers = 6
    step_per_worker = global_step // num_workers
    return A3CConfig(
        num_workers=num_workers,
        # have to use cpu for multiprocessing
        device=get_device("cpu"),
        total_steps=step_per_worker,
        rollout_len=32,
        learning_rate=1e-4,
        critic_lr=5e-5,
        critic_lr_gamma=0.95,
        gamma=0.99,
        gae_lambda_or_n_step=0.97,
        entropy_coef=LinearSchedule(start_e=0.2, end_e=0.1, duration=200),
        # entropy_coef=ConstantSchedule(0.1),
        value_loss_coef=0.02,
        max_grad_norm=0.5,
        normalize_returns=True,
        hidden_size=64,
        eval_episodes=50,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="CartPole-v1",
            vector_env_num=2,
            use_multi_processing=False,
            # max_steps=1000,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=A2CTrainer,
            agent_type=A2CAgent,
            output_dir="results/exercise6_a3c/cartpole/",
            save_result=True,
            model_filename="a3c.pth",
            repo_id="A3C-CartPoleV1",
            algorithm_name="A3C with GAE",
            extra_tags=("A3C", "GAE"),
        ),
    )


def get_env_for_play_and_hub(config: A3CConfig) -> EnvType:
    """Get the environment for play and hub."""
    _, eval_env = get_env_from_config(config.env_config)
    return eval_env
