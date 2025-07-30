import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from practice.base.context import ContextBase
from practice.base.trainer import TrainerBase
from practice.exercise2_dqn.dqn_exercise import BasicDQNPod, DQNConfig, DQNPod
from practice.exercise2_dqn.e22_double_dqn.double_dqn_exercise import DoubleDQNPod
from practice.exercise2_dqn.e24_rainbow.rainbow_exercise import RainbowPod
from practice.utils_for_coding.writer_utils import CustomWriter

POD_TYPE = {
    "basic": BasicDQNPod,
    "double": DoubleDQNPod,
    "rainbow": RainbowPod,
}


class DQNTrainer(TrainerBase):
    """A trainer for the DQN algorithm."""

    def __init__(self, config: DQNConfig, ctx: ContextBase) -> None:
        super().__init__(config=config, ctx=ctx)
        self._config: DQNConfig = config
        self._ctx: ContextBase = ctx

    def train(self) -> None:
        """Train the DQN agent with multiple environments."""
        # Initialize tensorboard writer
        writer = CustomWriter(
            track=self._ctx.track_and_evaluate,
            log_dir=self._config.artifact_config.get_tensorboard_dir(),
        )
        # Use environment from context - must be vector environment for DQN training
        envs = self._ctx.envs

        # Create trainer pod
        pod: DQNPod = POD_TYPE[self._config.dqn_algorithm](
            config=self._config, ctx=self._ctx, writer=writer
        )

        # Initialize environments
        states, _ = envs.reset()
        assert isinstance(states, np.ndarray), "States must be numpy array"
        # Track previous step terminal status to avoid invalid transitions
        prev_dones: NDArray[np.bool_] = np.zeros(envs.num_envs, dtype=np.bool_)
        episode_steps = 0

        # Training loop
        for step in tqdm(range(self._config.timesteps), desc="Training"):
            # Get actions for all environments
            actions = pod.action(states)
            # Step all environments
            next_states, rewards, terms, truncs, infos = envs.step(actions)

            # Cast rewards to numpy array for indexing
            rewards_np = np.asarray(rewards, dtype=np.float32)
            # Handle terminal observations and create proper training transitions
            dones = np.logical_or(terms, truncs, dtype=np.bool_)

            # Only store transitions for states that were not terminal in the previous step
            # we use AutoReset wrapper, so the envs will be reset automatically when it's done
            # when any done in n step, the next_states of n+1 step is the first of the next episode
            pre_non_terminal_mask = ~prev_dones
            if np.any(pre_non_terminal_mask):
                # Only store transitions where the previous step didn't end an episode
                pod.buffer_add(
                    states=states[pre_non_terminal_mask],
                    actions=actions[pre_non_terminal_mask],
                    rewards=rewards_np[pre_non_terminal_mask],
                    next_states=next_states[pre_non_terminal_mask],
                    dones=dones[pre_non_terminal_mask],
                )

            # Update state and previous done status for next iteration
            states = next_states
            prev_dones = dones

            # Training updates
            if step >= self._config.update_start_step:
                if step % self._config.train_interval == 0:
                    # sample batch and update
                    pod.update()

                if step % self._config.target_update_interval == 0:
                    pod.sync_target_net()

            # Log episode metrics
            episode_steps += writer.log_episode_stats_if_has(infos, episode_steps)

        # Cleanup
        writer.close()
