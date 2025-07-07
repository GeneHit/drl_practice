from typing import Any

import numpy as np
import torch


def change_state_to_tensor(state: Any) -> torch.Tensor:
    if isinstance(state, np.ndarray):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return torch.from_numpy(state).float()
    elif isinstance(state, torch.Tensor):
        return state
    else:
        raise ValueError("Invalid state type")
