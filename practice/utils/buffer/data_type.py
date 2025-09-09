from dataclasses import dataclass
from typing import Any, Sequence

import torch
from numpy.typing import NDArray
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class Experience:
    """The experience of the agent for one or a batch of steps.

    Attributes:
        states:      State at the start  [batch, state_dim]
        actions:     Action taken        [batch, action_dim]
        rewards:     Reward received     [batch]
        next_states: Next state          [batch, state_dim]
        dones:       Done mask (0/1)     [batch]
    """

    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "Experience":
        """Move all tensors to a device and/or dtype."""
        return Experience(
            states=self.states.to(device, dtype, non_blocking=True),
            actions=self.actions.to(device, dtype, non_blocking=True),
            rewards=self.rewards.to(device, dtype, non_blocking=True),
            next_states=self.next_states.to(device, dtype, non_blocking=True),
            dones=self.dones.to(device, dtype, non_blocking=True),
        )

    @classmethod
    def from_numpy(cls, **batch_data: NDArray[Any]) -> "Experience":
        """Create an Experience from numpy arrays."""
        return cls(
            states=torch.from_numpy(batch_data["states"]),
            actions=torch.from_numpy(batch_data["actions"]),
            rewards=torch.from_numpy(batch_data["rewards"]),
            next_states=torch.from_numpy(batch_data["next_states"]),
            dones=torch.from_numpy(batch_data["dones"]),
        )

    @classmethod
    def from_kwargs(cls, **batch_data: Tensor) -> "Experience":
        """Create an Experience from a dictionary of torch tensors."""
        return cls(
            states=batch_data["states"],
            actions=batch_data["actions"],
            rewards=batch_data["rewards"],
            next_states=batch_data["next_states"],
            dones=batch_data["dones"],
        )


def merge_experiences(exps: Sequence[Experience], cpu: bool = False) -> Experience:
    """Merge a sequence of experiences into one experience."""
    exps = list(exps)
    if not exps:
        raise ValueError("exps is empty")

    # collect all fields
    states, actions, rewards, next_states, dones = zip(
        *((e.states, e.actions, e.rewards, e.next_states, e.dones) for e in exps)
    )
    exp = Experience(
        states=torch.cat(list(states), dim=0),
        actions=torch.cat(list(actions), dim=0),
        rewards=torch.cat(list(rewards), dim=0),
        next_states=torch.cat(list(next_states), dim=0),
        dones=torch.cat(list(dones), dim=0),
    )
    if cpu:
        exp = Experience(
            states=exp.states.cpu(),
            actions=exp.actions.cpu(),
            rewards=exp.rewards.cpu(),
            next_states=exp.next_states.cpu(),
            dones=exp.dones.cpu(),
        )
    return exp
