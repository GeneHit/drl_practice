import time

import torch.distributed as dist

from practice.base.chest import AgentBase
from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.exercise1_q.q_table_exercise import QTable
from practice.utils.dist_utils import is_distributed
from practice.utils.evaluation_utils import evaluate_and_save_results
from practice.utils_for_coding.agent_utils import ACAgent, ContAgent, ContinuousAgent, NNAgent


def train_and_evaluate_network(config: BaseConfig, ctx: ContextBase) -> None:
    """Main training function for reinforcement learning algorithms."""
    trainer = config.artifact_config.trainer_type(config=config, ctx=ctx)

    start_time = time.time()
    trainer.train()
    train_duration_min = (time.time() - start_time) / 60

    # Create agent based on the configured agent type
    agent_type = config.artifact_config.agent_type

    # Check if this is a neural network agent
    if agent_type == QTable:
        assert isinstance(ctx.trained_target, QTable)
        agent: AgentBase = ctx.trained_target
    elif agent_type == NNAgent:
        agent = NNAgent(net=ctx.network)
    elif agent_type == ACAgent:
        agent = ACAgent(net=ctx.network)
    elif agent_type == ContinuousAgent:
        agent = ContinuousAgent(net=ctx.network)
    elif agent_type == ContAgent:
        agent = ContAgent(net=ctx.network)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    # Evaluate and save results
    if ctx.track_and_evaluate:
        evaluate_and_save_results(
            env=ctx.eval_env,
            agent=agent,
            config=config,
            meta_data={"train_duration_min": f"{train_duration_min:.2f}"},
        )

    if is_distributed():
        dist.destroy_process_group()
