import threading
from typing import Any

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from practice.utils.env_utils import extract_episode_data_from_infos


def log_episode_stats_if_has(
    writer: SummaryWriter,
    infos: dict[str, Any],
    episode_steps: int,
) -> int:
    """Log the episode stats if has episode data.

    Parameters:
    ----------
        writer: The writer.
        infos: The infos, which is the return value of the env.step()
        episode_steps: The episode steps.

    Returns:
    -------
        int: The number of ended episodes.
    """
    ep_rewards, ep_lengths = extract_episode_data_from_infos(infos)
    for idx, reward in enumerate(ep_rewards):
        writer.add_scalar("episode/reward", reward, episode_steps)
        writer.add_scalar("episode/length", ep_lengths[idx], episode_steps)
        episode_steps += 1

    return len(ep_rewards)


def log_action_stats(
    actions: Tensor,
    writer: SummaryWriter,
    step: int,
    data: dict[str, Tensor | float] = {},
    log_interval: int = 1,
    unblocked: bool = True,
) -> None:
    """Log the action stats.

    Args:
        actions: The actions.
        writer: The writer.
        step: The step.
        tensor_data: The data to log.
        metadata: The metadata to log.
        log_interval: The interval to log the stats.
        unblocked: Whether to log the stats in a separate thread.
    """

    def _log_scalar() -> None:
        writer.add_scalar("action/mean", actions.mean().item(), step)
        writer.add_scalar("action/std", actions.std().item(), step)
        for key, value in data.items():
            if isinstance(value, Tensor):
                writer.add_scalar(f"action/{key}", value.mean().item(), step)
            else:
                writer.add_scalar(f"action/{key}", value, step)

    if step % log_interval != 0:
        return

    # log the action stats
    if not unblocked:
        _log_scalar()
        return
    # log the action stats in a separate thread
    thread = threading.Thread(target=_log_scalar, daemon=True)
    thread.start()


def log_stats(
    data: dict[str, Tensor | float],
    writer: SummaryWriter,
    step: int,
    log_interval: int = 1,
    unblocked: bool = False,
) -> None:
    """Log the stats.

    Args:
        data: The data to log.
        writer: The writer.
        step: The step.
        unblocked: Whether to log the stats in a separate thread.
    """

    def _log_scalar() -> None:
        for key, value in data.items():
            # to simply the code, always call mean()
            if isinstance(value, Tensor):
                writer.add_scalar(f"{key}", value.mean().item(), step)
            else:
                writer.add_scalar(f"{key}", value, step)

    if step % log_interval != 0:
        return

    # log the stats
    if not unblocked:
        _log_scalar()
        return
    # log the update stats in a separate thread
    thread = threading.Thread(target=_log_scalar, daemon=True)
    thread.start()
