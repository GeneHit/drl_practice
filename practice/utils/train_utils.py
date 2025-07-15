import numpy as np

from practice.base.chest import AgentBase
from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.exercise1_q.q_table_exercise import QTable
from practice.utils.evaluation_utils import evaluate_and_save_results
from practice.utils_for_coding.agent_utils import NNAgent


def train_and_evaluate_network(config: BaseConfig, ctx: ContextBase) -> None:
    """Main training function for reinforcement learning algorithms."""
    trainer = config.artifact_config.trainer_type(config=config, ctx=ctx)
    trainer.train()

    # Create agent based on the configured agent type
    agent_type = config.artifact_config.agent_type

    # Check if this is a neural network agent
    if agent_type == NNAgent:
        agent: AgentBase = NNAgent(net=ctx.network)
    elif agent_type == QTable:
        assert isinstance(ctx.trained_target, np.ndarray)
        agent = QTable(q_table=ctx.trained_target)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    # Evaluate and save results
    evaluate_and_save_results(env=ctx.eval_env, agent=agent, config=config)
