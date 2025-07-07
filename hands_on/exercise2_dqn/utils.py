from typing import Any

import gymnasium as gym


def describe_wrappers(env: gym.Env[Any, Any]) -> list[str]:
    stack = []
    while hasattr(env, "env"):
        stack.append(type(env).__name__)
        env = env.env
    stack.append(type(env).__name__)  # base env
    return list(reversed(stack))
