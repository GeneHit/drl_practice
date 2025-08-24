from dataclasses import dataclass
from typing import Any

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
            states=self.states.to(device, dtype),
            actions=self.actions.to(device, dtype),
            rewards=self.rewards.to(device, dtype),
            next_states=self.next_states.to(device, dtype),
            dones=self.dones.to(device, dtype),
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
    def from_dict(cls, **batch_data: Tensor) -> "Experience":
        """Create an Experience from a dictionary of torch tensors."""
        return cls(
            states=batch_data["states"],
            actions=batch_data["actions"],
            rewards=batch_data["rewards"],
            next_states=batch_data["next_states"],
            dones=batch_data["dones"],
        )
