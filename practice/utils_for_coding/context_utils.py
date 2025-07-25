from dataclasses import dataclass

import torch
import torch.nn as nn

from practice.base.context import ContextBase


@dataclass(frozen=True, kw_only=True)
class ACContext(ContextBase):
    """The context for the Actor-Critic separatedly.

    Separated network&optimizer for actor/critic. The network&optimizer in ContextBase is for actor.
    """

    critic: nn.Module
    """The critic network."""

    critic_optimizer: torch.optim.Optimizer
    """The optimizer for the critic."""
