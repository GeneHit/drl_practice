from practice.exercise4_curiosity.enhanced_reinforce import (
    EnhancedReinforceConfig,
    ReinforceContext,
)
from practice.utils.train_utils import train_and_evaluate_network

# def reinforce_train(
#     config: EnhancedReinforceConfig,
#     ctx: ReinforceContext,
# ) -> None:
#     """Main training function for single-environment REINFORCE."""
#     trainer = EnhancedReinforceTrainer(config=config, ctx=ctx)
#     trainer.train()

#     # Create agent and evaluate/save results on a single environment
#     evaluate_and_save_results(env=ctx.eval_env, agent=NNAgent(net=ctx.network), config=config)


def curiosity_train(
    config: EnhancedReinforceConfig,
    ctx: ReinforceContext,
) -> None:
    train_and_evaluate_network(config=config, ctx=ctx, Trainer=ctx.trainer_name)
