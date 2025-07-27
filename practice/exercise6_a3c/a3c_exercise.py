import time
from dataclasses import dataclass

from gymnasium.spaces import Discrete
from torch.multiprocessing import Process
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from practice.base.context import ContextBase
from practice.exercise5_a2c.a2c_gae_exercise import A2CConfig, A2CTrainer, ActorCritic
from practice.utils.env_utils import get_env_from_config
from practice.utils.evaluation_utils import evaluate_and_save_results
from practice.utils_for_coding.agent_utils import ACAgent


@dataclass(frozen=True, kw_only=True)
class A3CConfig(A2CConfig):
    """The configuration for the A3C algorithm."""

    num_workers: int
    """The number of workers."""


def worker_process(
    worker_id: int,
    config: A3CConfig,
    actor_critic: ActorCritic,
) -> None:
    """The worker process for the A3C algorithm.

    The worker process will:
    1. Initialization:
        local optimizer/scheduler for global actor-critic network.
        local environment.
    2. Training with A2CTrainer:
        - collect data from the local environment.
        - update the global actor-critic network.

    Parameters
    ----------
    worker_id: int
        The id of the worker.
    config: A3CConfig
        The configuration for the A3C algorithm.
    actor_critic: nn.Module
        The global shared actor-critic network.
    """
    # local optimizer
    policy_and_shared_params = list(actor_critic.shared_layers.parameters()) + list(
        actor_critic.policy_logits.parameters()
    )
    optimizer = Adam(
        [
            {"params": policy_and_shared_params, "lr": config.learning_rate},
            {"params": actor_critic.value_head.parameters(), "lr": config.critic_lr},
        ]
    )
    # local scheduler
    lr_schedulers: tuple[LRScheduler, ...] = ()
    critic_lr_gamma = config.critic_lr_gamma
    if critic_lr_gamma is not None:
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=[
                lambda epoch: 1.0,  # group 0: policy and shared lr
                lambda epoch: critic_lr_gamma**epoch,  # group 1: critic lr
            ],
        )
        lr_schedulers = (scheduler,)
    # local environment
    train_env, unused_eval_env = get_env_from_config(config.env_config)
    ctx = ContextBase(
        train_env=train_env,
        eval_env=unused_eval_env,
        trained_target=actor_critic,
        optimizer=optimizer,
        lr_schedulers=lr_schedulers,
    )

    # Training
    try:
        trainer = A2CTrainer(config, ctx, log_prefix=f"{worker_id}")
        trainer.train()
    except Exception as e:
        print(f"Worker {worker_id} failed with error: {e}")
    finally:
        train_env.close()
        unused_eval_env.close()


def a3c_train(config: A3CConfig) -> None:
    """The main function for the A3C algorithm

    The A3C algorithm will:
    1. Initialize the global shared actor-critic network.
    2. Training with Asynchronous workers (A2CTrainer).
    3. Evaluation and save results.
    """
    unused_train_env, eval_env = get_env_from_config(config.env_config)
    unused_train_env.close()
    del unused_train_env

    # Initialize the global shared actor-critic network.
    obs_shape = eval_env.observation_space.shape
    assert obs_shape is not None
    assert isinstance(eval_env.action_space, Discrete)
    action_n = int(eval_env.action_space.n)
    actor_critic = ActorCritic(
        obs_dim=obs_shape[0],
        n_actions=action_n,
        hidden_size=config.hidden_size,
    )
    actor_critic.to(config.device)
    actor_critic.share_memory()

    # Asynchronous workers and training
    start_time = time.time()
    if config.num_workers == 1:
        worker_process(0, config, actor_critic)
    else:
        processes = []
        for worker_id in range(config.num_workers):
            p = Process(target=worker_process, args=(worker_id, config, actor_critic))
            p.start()
            processes.append(p)
        # wait for all workers to finish
        for p in processes:
            p.join()

    # Evaluation and save results
    train_duration_min = (time.time() - start_time) / 60
    agent_type = config.artifact_config.agent_type
    assert agent_type == ACAgent
    agent = ACAgent(net=actor_critic)
    evaluate_and_save_results(
        env=eval_env,
        agent=agent,
        config=config,
        meta_data={"train_duration_min": f"{train_duration_min:.2f}"},
    )
    eval_env.close()
