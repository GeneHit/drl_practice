from typing import Type

from hands_on.utils.agent_utils import NNAgent
from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.trainer import TrainerBase
from practice.utils.evaluation_utils import evaluate_and_save_results


def train_and_evaluate_network(
    config: BaseConfig,
    ctx: ContextBase,
    Trainer: Type[TrainerBase],
) -> None:
    """Main training function for single-environment REINFORCE."""
    trainer = Trainer(config=config, ctx=ctx)
    trainer.train()

    # Create agent and evaluate/save results on a single environment
    evaluate_and_save_results(env=ctx.eval_env, agent=NNAgent(net=ctx.network), config=config)
