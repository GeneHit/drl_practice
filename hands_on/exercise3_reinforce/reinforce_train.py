from typing import Any, Callable

from hands_on.exercise3_reinforce.reinforce_exercise import (
    EnvsType,
    EnvType,
)


def reinforce_train_with_envs(
    envs: EnvsType,
    env_fn: Callable[[], EnvType],
    cfg_data: dict[str, Any],
) -> None:
    raise NotImplementedError


def main(cfg_data: dict[str, Any]) -> None:
    raise NotImplementedError
