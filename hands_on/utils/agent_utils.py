from typing import Any

import torch
import torch.nn as nn
from numpy.typing import NDArray

from hands_on.base import ActType, AgentBase
from hands_on.utils_for_coding.numpy_tensor_utils import (
    get_tensor_expanding_axis,
)


class NNAgent(AgentBase):
    """Agent that uses a neural network to make decisions.

    Only use for evaluation/gameplay, not for training.
    """

    def __init__(
        self,
        net: nn.Module,
    ):
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
        """Save the DQN model."""
        assert pathname.endswith(".pth")
        torch.save(self._net, pathname)

    @classmethod
    def load_from_checkpoint(
        cls, pathname: str, device: torch.device | None
    ) -> "NNAgent":
        """Load the DQN model."""
        assert pathname.endswith(".pth")
        net = torch.load(pathname, map_location=device, weights_only=False)
        net = net.to(device)
        return cls(net=net)
