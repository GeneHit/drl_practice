from typing import Any

import numpy as np
import torch


def get_tensor_expanding_axis(state: np.typing.NDArray[Any], axis: int = 0) -> torch.Tensor:
    return torch.from_numpy(np.expand_dims(state, axis=axis)).float()
