from typing import Any

import torch
import torch.nn as nn
from numpy.typing import NDArray

from practice.base.chest import AgentBase
from practice.base.env_typing import ActType
from practice.utils_for_coding.numpy_tensor_utils import get_tensor_expanding_axis


class NNAgent(AgentBase):
    """Agent that uses a neural network to make decisions.

    Only use for evaluation/gameplay, not for training.
    """

    def __init__(self, net: nn.Module) -> None:
        self._net = net
        self._device = next(net.parameters()).device

    def action(self, state: NDArray[Any]) -> ActType:
        """Get action for single state using greedy policy."""
        # Always use greedy policy for trained agent evaluation
        state_tensor = get_tensor_expanding_axis(state).to(self._device)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(state_tensor).cpu()
        return ActType(logits.argmax().item())

    def only_save_model(self, pathname: str) -> None:
        """Save the NN model."""
        _save_model(self._net, pathname)

    @classmethod
    def load_from_checkpoint(cls, pathname: str, device: torch.device | None) -> "NNAgent":
        """Load the NN model."""
        return cls(net=_load_model(pathname, device))


class A2CAgent(AgentBase):
    """Agent that uses a neural network to make decisions.

    Only use for evaluation/gameplay, not for training.
    """

    def __init__(self, net: nn.Module) -> None:
        self._net = net
        self._device = next(net.parameters()).device

    def action(self, state: NDArray[Any]) -> ActType:
        """Get action for single state using greedy policy."""
        # Always use greedy policy for trained agent evaluation
        state_tensor = get_tensor_expanding_axis(state).to(self._device)
        self._net.eval()
        with torch.no_grad():
            policy_logits_or_value = self._net(state_tensor)

            if isinstance(policy_logits_or_value, tuple):
                # returning (policy_logits, value)
                policy_logits, _ = policy_logits_or_value
            else:
                # returning policy_logits
                policy_logits = policy_logits_or_value

            # get max probability action
            return ActType(policy_logits.argmax(dim=-1).cpu().item())

    def only_save_model(self, pathname: str) -> None:
        """Save the A2C model."""
        _save_model(self._net, pathname)

    @classmethod
    def load_from_checkpoint(cls, pathname: str, device: torch.device | None) -> "A2CAgent":
        """Load the A2C model."""
        return cls(net=_load_model(pathname, device))


def _save_model(net: nn.Module, pathname: str) -> None:
    """Save the model."""
    assert pathname.endswith(".pth")
    torch.save(net, pathname)


def _load_model(pathname: str, device: torch.device | None) -> nn.Module:
    """Load the model."""
    assert pathname.endswith(".pth")
    net: nn.Module = torch.load(pathname, map_location=device, weights_only=False).to(device)
    return net
