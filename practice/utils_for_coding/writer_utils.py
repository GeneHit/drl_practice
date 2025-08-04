from concurrent.futures import ThreadPoolExecutor
from typing import Any

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from practice.utils.env_utils import extract_episode_data_from_infos


class CustomWriter:
    """Custom writer for logging the stats.

    Args:
        log_dir: The directory to log the stats.
        track: Whether to track the stats.
        max_workers: The maximum number of workers to use.
    """

    def __init__(self, track: bool, log_dir: str = "", max_workers: int = 4) -> None:
        self._track = track
        self._writer: SummaryWriter | None = None
        self._executor: ThreadPoolExecutor | None = None
        if track:
            assert log_dir, "log_dir must be provided when track is True"
            self._writer = SummaryWriter(log_dir)
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def __del__(self) -> None:
        """Close the writer."""
        self.close()

    @property
    def writer(self) -> SummaryWriter:
        """The writer."""
        assert self._writer is not None  # make mypy happy
        return self._writer

    def log_episode_stats_if_has(
        self, infos: dict[str, Any], episode_steps: int, log_interval: int = 1
    ) -> int:
        """Log the episode stats if has episode data.

        If the writer is not tracked, this method does nothing.

        Args:
            infos: The infos, which is the return value of the env.step()
            episode_steps: The episode steps.
            log_interval: The interval to log the stats.

        Returns:
            int: The number of ended episodes.
        """
        ep_rewards, ep_lengths = extract_episode_data_from_infos(infos)
        if not self._track:
            return len(ep_rewards)

        for idx, reward in enumerate(ep_rewards):
            self.log_stats(
                data={
                    "episode/reward": reward,
                    "episode/length": ep_lengths[idx],
                },
                step=episode_steps,
                log_interval=log_interval,
                blocked=True,
            )
            episode_steps += 1
        return len(ep_rewards)

    def log_action_stats(
        self,
        actions: Tensor,
        step: int,
        data: dict[str, Tensor | float] = {},
        log_interval: int = 1,
        blocked: bool = True,
    ) -> None:
        """Log the action stats.

        If the writer is not tracked, this method does nothing.

        Args:
            actions: The actions.
            step: The step.
            data: The data to log.
            log_interval: The interval to log the stats.
            blocked: Whether to log the stats in a separate thread.
        """
        if not self._track or step % log_interval != 0:
            return

        def _log_scalar() -> None:
            self.writer.add_scalar("action/mean", actions.cpu().mean().item(), step)
            self.writer.add_scalar("action/std", actions.cpu().std().item(), step)
            for key, value in data.items():
                if isinstance(value, Tensor):
                    self.writer.add_scalar(f"action/{key}", value.cpu().mean().item(), step)
                else:
                    self.writer.add_scalar(f"action/{key}", value, step)

        if blocked:
            _log_scalar()
            return
        assert self._executor is not None  # make mypy happy
        self._executor.submit(_log_scalar)

    def log_stats(
        self,
        data: dict[str, Tensor | float | int],
        step: int,
        log_interval: int = 1,
        blocked: bool = True,
    ) -> None:
        """Log the stats.

        If the writer is not tracked, this method does nothing.

        Args:
            data: The data to log.
            step: The step.
            log_interval: The interval to log the stats.
            blocked: Whether to log the stats in a separate thread.
        """
        if not self._track or step % log_interval != 0:
            return

        def _log_scalar() -> None:
            for key, value in data.items():
                if isinstance(value, Tensor):
                    self.writer.add_scalar(f"{key}", value.cpu().mean().item(), step)
                else:
                    self.writer.add_scalar(f"{key}", value, step)

        if blocked:
            _log_scalar()
            return
        assert self._executor is not None  # make mypy happy
        self._executor.submit(_log_scalar)

    def close(self) -> None:
        """Close the writer."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        if self._writer is not None:
            self._writer.close()
