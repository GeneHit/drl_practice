from hands_on.utils.agent_utils import NNAgent
from practice.exercise4_curiosity.enhanced_reinforce import (
    EnhancedReinforceConfig,
    EnhancedReinforceTrainer,
    ReinforceContext,
)
from practice.utils.evaluation_utils import evaluate_and_save_results


def reinforce_train(
    config: EnhancedReinforceConfig,
    ctx: ReinforceContext,
) -> None:
    """Main training function for single-environment REINFORCE."""
    trainer = EnhancedReinforceTrainer(config=config, ctx=ctx)
    trainer.train()

    # Create agent and evaluate/save results on a single environment
    evaluate_and_save_results(env=ctx.eval_env, agent=NNAgent(net=ctx.network), config=config)
