from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from practice.base.context import ContextBase
from practice.base.env_typing import ActTypeC, ObsType
from practice.base.trainer import TrainerBase
from practice.exercise9_sac.sac_exercise import SACConfig, _SACPod
from practice.exercise12_mbpo.model_based_env import EnvModel, ModelBasedConfig, ModelBasedEnv
from practice.utils_for_coding.buffer import Experience, ReplayBuffer
from practice.utils_for_coding.buffer.data_type import merge_experiences
from practice.utils_for_coding.context_utils import ACContext
from practice.utils_for_coding.scheduler_utils import ScheduleBase
from practice.utils_for_coding.writer_utils import CustomWriter


@dataclass(kw_only=True, frozen=True)
class MBPOConfig(SACConfig):
    """The configuration for the MBPO algorithm."""

    train_interval: int
    """The interval of training the env model and SAC."""

    rollout_num: int
    """The rollout number of the model-based environment every epoch."""

    rollout_len: ScheduleBase
    """The generated rollout length of the model-based environment every epoch."""

    update_num_per_epoch: int
    """The update number of the SAC every epoch."""

    batch_rate_of_model_sample: ScheduleBase
    """The batch rate of the model sample."""

    model_based_config: ModelBasedConfig
    """The configuration for the model-based environment."""

    model_replay_buffer_capacity: int
    """The capacity of the model replay buffer."""


@dataclass(kw_only=True, frozen=True)
class MBPOContext(ContextBase):
    """The context for the MBPO algorithm."""

    critic: torch.nn.Module
    """The critic network."""

    critic_optimizer: torch.optim.Optimizer
    """The optimizer for the critic."""

    env_model: EnvModel
    """The environment model."""


class MBPOTrainer(TrainerBase):
    """A trainer for the MBPO algorithm."""

    def __init__(self, config: MBPOConfig, ctx: MBPOContext) -> None:
        super().__init__(config=config, ctx=ctx)
        self._config: MBPOConfig = config
        self._ctx: MBPOContext = ctx

    def train(self) -> None:
        """Train the policy network with a vectorized environment.

        Steps:
        for epoch in 1...epochs:
            1. step the real environment for real_step_per_epoch steps, and buffer the replay
            2. update env model, model replay buffer and SAC if necessary
                - 2.1 train all env models with real replay buffer
                - 2.2 use random model to generate rollout
                    - 2.2.1 get rollout_num samples from real replay
                    - 2.2.2 run rollout_len(epoch) steps for each sample and buffer it
                - 2.3 train the SAC with mixed data in update_num_per_epoch steps
                    - 2.3.1 sample from the model replay with changed batch_rate_of_model_sample
                    - 2.3.2 sample from the real replay with (1 - batch_rate_of_model_sample)
                    - 2.3.3 train the SAC with mixed data
        """
        # Initialize tensorboard writer
        writer = CustomWriter(
            track=self._ctx.track_and_evaluate,
            log_dir=self._config.artifact_config.get_tensorboard_dir(),
        )
        # Use environment from context - must be vector environment
        envs = self._ctx.continuous_envs

        # Create trainer pod
        pod = _MBPOPod(config=self._config, ctx=self._ctx, writer=writer)
        # Create replay buffer using observation shape from context
        obs_dtype = envs.single_observation_space.dtype
        assert obs_dtype in (np.float32, np.uint8)
        assert envs.single_action_space.dtype == np.float32
        env_buffer = ReplayBuffer(capacity=self._config.replay_buffer_capacity)

        # Initialize environments
        states, _ = envs.reset()
        assert isinstance(states, np.ndarray), "States must be numpy array"
        # Track previous step terminal status to avoid invalid transitions
        prev_dones: NDArray[np.bool_] = np.zeros(envs.num_envs, dtype=np.bool_)
        episode_steps = 0

        # loop
        timestep = self._config.total_steps // envs.num_envs
        start_step = self._config.update_start_step // envs.num_envs
        for step in tqdm(range(timestep), desc="Training"):
            # Get actions for all environments
            actions = pod.action(states, step)
            # Step the environment
            next_states, rewards, terminated, truncated, infos = envs.step(actions)

            # Cast rewards to numpy array for indexing
            rewards = np.asarray(rewards, dtype=np.float32)
            # Handle terminal observations and create proper training transitions
            dones = np.logical_or(terminated, truncated, dtype=np.bool_)

            # Only store transitions for states that were not terminal in the previous step
            # we use AutoReset wrapper, so the envs will be reset automatically when it's done
            # when any done in n step, the next_states of n+1 step is the first of the next episode
            pre_non_terminal_mask = ~prev_dones
            if np.any(pre_non_terminal_mask):
                # Create training transitions
                env_buffer.add_batch(
                    states=states[pre_non_terminal_mask],
                    actions=actions[pre_non_terminal_mask],
                    rewards=rewards[pre_non_terminal_mask],
                    next_states=next_states[pre_non_terminal_mask],
                    dones=dones[pre_non_terminal_mask],
                )

            states = next_states
            prev_dones = dones
            # Log episode metrics
            episode_steps += writer.log_episode_stats_if_has(infos, episode_steps)

            # Training updates
            if step >= start_step and step % self._config.train_interval == 0:
                if len(env_buffer) < self._config.batch_size:
                    continue

                # update env model, model replay buffer and SAC
                pod.update(env_buffer=env_buffer, step=step)

        writer.close()


class _MBPOPod:
    """A pod for the MBPO algorithm."""

    def __init__(self, config: MBPOConfig, ctx: MBPOContext, writer: CustomWriter) -> None:
        self._config = config
        self._ctx = ctx
        self._writer = writer

        self._model_buffer = ReplayBuffer(capacity=config.model_replay_buffer_capacity)
        self._model_env = ModelBasedEnv(model=ctx.env_model, cfg=config.model_based_config)
        self._sac_pod = _SACPod(
            config=config,
            ctx=ACContext(
                train_env=ctx.train_env,
                eval_env=ctx.eval_env,
                trained_target=ctx.trained_target,
                optimizer=ctx.optimizer,
                critic=ctx.critic,
                critic_optimizer=ctx.critic_optimizer,
                lr_schedulers=ctx.lr_schedulers,
                track_and_evaluate=ctx.track_and_evaluate,
            ),
            writer=writer,
        )

    def action(self, state: NDArray[ObsType], step: int) -> NDArray[ActTypeC]:
        """Get actions for all environments."""
        return self._sac_pod.action(state=state, step=step)

    def update(self, env_buffer: ReplayBuffer, step: int) -> None:
        """Update the env model, model replay buffer and SAC.

        Steps:
        1. train all env models with real replay buffer
        2. use random model to generate rollout
            - 2.1 get rollout_num samples from real replay
            - 2.2 run rollout_len(epoch) steps for each sample
            - 2.3 buffer the model replay
        3. train the SAC with mixed data in update_num_per_epoch steps
            - 3.1 sample from the model replay with changed batch_rate_of_model_sample
            - 3.2 sample from the real replay with (1 - batch_rate_of_model_sample)
            - 3.3 train the SAC with mixed data

        Args:
            env_buffer: The experience from real data.
            step: The current step.
        """
        # 1. train all env models
        loss_stats = self._model_env.train(buffer=env_buffer)
        self._writer.log_stats(
            data={"model_loss/" + k: v[-1] for k, v in loss_stats.items()},
            step=step,
            log_interval=self._config.log_interval,
            blocked=False,
        )

        # 2. use random model to generate rollout and buffer it
        rollouts = self._generate_rollouts(
            states=env_buffer.sample(self._config.rollout_num).states, step=step
        )
        self._model_buffer.add_experience(rollouts)

        # 3. train the SAC with mixed data
        model_data_num = int(
            self._config.batch_size * self._config.batch_rate_of_model_sample(step)
        )
        real_data_num = self._config.batch_size - model_data_num
        for _ in range(self._config.update_num_per_epoch):
            model_data = self._model_buffer.sample(model_data_num)
            real_data = env_buffer.sample(real_data_num)
            mixed_data = merge_experiences([model_data, real_data])

            self._sac_pod.update(experience=mixed_data, step=step)

    def _generate_rollouts(self, states: torch.Tensor, step: int) -> Experience:
        """Generate rollouts.

        Args:
            exp: The initial experience to generate rollouts.
            step: The current step.

        Returns:
            The rollouts.
        """
        rollout_len = int(self._config.rollout_len(step))
        rollouts: list[Experience] = []
        rollout_num = states.shape[0]

        for i in range(rollout_num):
            state = states[i : i + 1]
            self._model_env.set_rollout_model()

            for _ in range(rollout_len):
                action = self._sac_pod.action_torch(state=state)
                next_state, reward, done = self._model_env.step(state, action)
                rollouts.append(
                    Experience(
                        states=state,
                        actions=action,
                        rewards=reward,
                        next_states=next_state,
                        dones=done,
                    )
                )

                state = next_state
                if done.squeeze(-1).any().item():
                    break

        return merge_experiences(rollouts)
