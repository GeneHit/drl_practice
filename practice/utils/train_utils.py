import time

import torch.distributed as dist
from torch import nn

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.exercise1_q.q_table_exercise import QTable
from practice.utils.dist_utils import is_distributed
from practice.utils.eval_utils import evaluate_and_save_results


def train_and_evaluate_network(config: BaseConfig, ctx: ContextBase) -> None:
    """Main training function for reinforcement learning algorithms."""
    trainer = config.artifact_config.trainer_type(config=config, ctx=ctx)

    start_time = time.time()
    trainer.train()
    train_duration_min = (time.time() - start_time) / 60

    # Check if this is a neural network agent
    if isinstance(ctx.trained_target, QTable):
        # the trained target is a q-table
        agent: nn.Module | QTable = ctx.table
    else:
        # the trained target is a neural network
        agent = ctx.network

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
