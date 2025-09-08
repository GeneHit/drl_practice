from typing import Any, TypeVar, Union

import numpy as np
import torch

T = TypeVar("T", bound=Union[np.floating[Any], np.integer[Any]])
INT_T = TypeVar("INT_T", bound=np.integer[Any])


def get_tensor_expanding_axis(state: np.typing.NDArray[Any], axis: int = 0) -> torch.Tensor:
    return torch.from_numpy(np.expand_dims(state, axis=axis)).float()


def as_tensor_on(val: float | int, ref_tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(val, device=ref_tensor.device, dtype=ref_tensor.dtype)


def np2tensor(x: np.typing.NDArray[Any], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a numpy array to a tensor."""
    return torch.from_numpy(x).to(dtype=dtype)


def tensor2np_1d(x: torch.Tensor, dtype: T) -> np.typing.NDArray[T]:
    """Convert a tensor to a 1D numpy array."""
    return x.cpu().numpy().reshape(-1).astype(dtype)


def argmax_action(x: torch.Tensor, dtype: type[INT_T]) -> INT_T:
    """Get the argmax actionof a tensor on CPU.

    Equivalent to the greedy strategy.
    """
    return dtype(x.argmax(dim=-1).cpu().item())
