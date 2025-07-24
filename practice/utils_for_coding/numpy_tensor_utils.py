from typing import Any

import numpy as np
import torch


def get_tensor_expanding_axis(state: np.typing.NDArray[Any], axis: int = 0) -> torch.Tensor:
    return torch.from_numpy(np.expand_dims(state, axis=axis)).float()


def as_tensor_on(val: float | int, ref_tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(val, device=ref_tensor.device, dtype=ref_tensor.dtype)
