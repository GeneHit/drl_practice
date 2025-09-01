from abc import ABC, abstractmethod
from typing import Any, Iterator

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset


class BufferBase(ABC):
    """Abstract base class for buffers with common functionality."""

    def __init__(self, capacity: int) -> None:
        """
        Initialize the buffer with a given capacity.

        Args:
            capacity: Maximum number of samples the buffer can hold
        """
        self._capacity = capacity
        self._ptr = 0  # current write pointer
        self._size = 0  # current valid data size
        self._initialized = False

    def clear(self) -> None:
        """Clear the buffer."""
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self._size

    @property
    def capacity(self) -> int:
        """Return the capacity of the buffer."""
        return self._capacity

    def _write_batch_common(self, batch_size: int) -> tuple[int, int]:
        """Common logic for batch writing - returns ptr_end and updates internal state.

        Args:
            batch_size: Size of the batch to write

        Returns:
            Tuple of (ptr_end, old_ptr) for use by concrete implementations
        """
        ptr_end = self._ptr + batch_size
        old_ptr = self._ptr

        # Update pointer and size
        self._ptr = ptr_end % self._capacity
        self._size = min(self._size + batch_size, self._capacity)

        return ptr_end, old_ptr

    def _validate_index(self, idx: int) -> int:
        """Validate index."""
        if not self._initialized:
            raise RuntimeError("Buffer not initialized. Call add_batch first.")

        if idx < 0:
            idx = self._size + idx
        if idx < 0 or idx >= self._size:
            raise IndexError(f"Index {idx} out of range for buffer of size {self._size}")
        return idx

    @abstractmethod
    def add_batch(self, **batch_data: Any) -> Any:
        """Add a batch of data to the buffer."""

    @abstractmethod
    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample a random batch from the buffer."""

    @abstractmethod
    def sample_by_idxs(self, idxs: NDArray[np.int64] | Tensor) -> dict[str, Any]:
        """Sample data by specific indices."""

    def dataloader(
        self,
        batch_size: int,
        *,
        ratio: float = 1.0,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> DataLoader[dict[str, Tensor]]:
        """Get a standard torch.utils.data.DataLoader of all data in the buffer."""
        device = getattr(self, "_device", torch.device("cpu"))
        is_cuda = device.type == "cuda"

        # CUDA tensors cannot be pickled between processes
        if is_cuda and num_workers != 0:
            raise RuntimeError(
                "Buffer stores CUDA tensors: please set num_workers=0 for dataloader."
            )
        if len(self) == 0:
            raise RuntimeError("Cannot create dataloader from empty buffer. Add data first.")

        class BufferIterable(IterableDataset[dict[str, Tensor]]):
            def __init__(self, buf: "BufferBase") -> None:
                self._buf = buf

            def __iter__(self) -> Iterator[dict[str, Tensor]]:
                n = len(self._buf)
                if ratio < 1.0:
                    n = int(n * ratio)
                if shuffle:
                    order = torch.randperm(n, device=device, dtype=torch.int64)
                else:
                    order = torch.arange(n, device=device, dtype=torch.int64)
                for s in range(0, n, batch_size):
                    e = s + batch_size
                    if e > n and drop_last:
                        break
                    batch = self._buf.sample_by_idxs(order[s:e])
                    if isinstance(next(iter(batch.values())), Tensor):
                        yield batch
                    else:
                        yield {key: torch.from_numpy(arr) for key, arr in batch.items()}

        return DataLoader(
            BufferIterable(self),
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )


class BufferNP(BufferBase):
    """Buffer implementation for numpy arrays."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._data: dict[str, NDArray[Any]] = {}

    def add_batch(self, **batch_data: NDArray[Any]) -> NDArray[np.int64]:
        """Add a batch of numpy arrays to the buffer.

        Args:
            **batch_data: Keyword arguments containing numpy arrays

        Returns:
            Indices where the data was written
        """
        if not batch_data:
            return np.array([], dtype=np.int64)

        batch_size = next(iter(batch_data.values())).shape[0]
        if batch_size == 0:
            return np.array([], dtype=np.int64)

        # Initialize storage when first called
        self._init_memory_if_necessary(**batch_data)

        # Batch writing: handle both no wrap-around and wrap-around cases
        ptr_end, old_ptr = self._write_batch_common(batch_size)

        if ptr_end <= self._capacity:
            # No wrap-around writing
            return self._write_no_wraparound(batch_data, old_ptr, ptr_end)
        else:
            # Wrap-around writing
            return self._write_wraparound(batch_data, batch_size, old_ptr)

    def sample(self, batch_size: int) -> dict[str, NDArray[Any]]:
        """Sample a random batch from the buffer.

        Args:
            batch_size: Number of samples to return

        Returns:
            dictionary containing sampled numpy arrays
        """
        assert self._size >= batch_size > 0, f"Buffer size {self._size}, but required {batch_size}"
        idxs = np.random.randint(0, self._size, (batch_size,), dtype=np.int64)
        return self.sample_by_idxs(idxs)

    def sample_by_idxs(self, idxs: NDArray[np.int64] | Tensor) -> dict[str, NDArray[Any]]:
        """Sample data by specific indices.

        Args:
            idxs: Numpy array or torch tensor of indices to sample

        Returns:
            dictionary containing sampled numpy arrays
        """
        assert self._initialized, "Memory not initialized. Call add_batch first."

        # Convert torch tensor to numpy array if necessary
        if isinstance(idxs, Tensor):
            idxs = idxs.cpu().numpy()

        # Check if all indices are within valid range
        if np.any((idxs >= self._size) | (idxs < 0)):
            raise ValueError(
                f"Invalid indices: indices must be in range [0, {self._size}), got {idxs}"
            )

        return {key: arr[idxs] for key, arr in self._data.items()}

    def _write_no_wraparound(
        self, batch_data: dict[str, NDArray[Any]], start: int, end: int
    ) -> NDArray[np.int64]:
        """Write data without wrap-around."""
        for key, data in batch_data.items():
            self._data[key][start:end] = data

        return np.arange(start, end, dtype=np.int64)

    def _write_wraparound(
        self, batch_data: dict[str, NDArray[Any]], batch_size: int, ptr: int
    ) -> NDArray[np.int64]:
        """Write data with wrap-around."""
        head_size = self._capacity - ptr
        tail_size = batch_size - head_size

        # Write data to each field
        for key, data in batch_data.items():
            self._data[key][ptr:] = data[:head_size]
            self._data[key][:tail_size] = data[head_size:]

        # Return written indices
        head_indices = np.arange(ptr, self._capacity, dtype=np.int64)
        tail_indices = np.arange(0, tail_size, dtype=np.int64)
        return np.concatenate([head_indices, tail_indices])

    def _init_memory_if_necessary(self, **batch_data: NDArray[Any]) -> None:
        """Initialize memory allocation based on the first batch of data.

        Args:
            **batch_data: Keyword arguments containing numpy arrays
        """
        if self._initialized:
            return

        # Pre-allocate memory for each field
        for key, data in batch_data.items():
            shape = (self._capacity, *data.shape[1:])
            self._data[key] = np.empty(shape, dtype=data.dtype)

        self._initialized = True


class BufferTorch(BufferBase):
    """Buffer implementation for torch tensors.

    All tensors are stored on the same device.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._data: dict[str, Tensor] = {}
        # Note: all tensors are stored on the same device. cpu will be overwritten during init.
        self._device: torch.device = torch.device("cpu")

    def add_batch(self, **batch_data: Tensor) -> Tensor:
        """Add a batch of torch tensors to the buffer.

        Args:
            **batch_data: Keyword arguments containing torch tensors

        Returns:
            Indices where the data was written
        """
        if not batch_data:
            return torch.tensor([], dtype=torch.int64)

        batch_size = next(iter(batch_data.values())).shape[0]
        if batch_size == 0:
            return torch.tensor([], dtype=torch.int64)

        # Initialize storage when first called
        self._init_memory_if_necessary(**batch_data)

        # Batch writing: handle both no wrap-around and wrap-around cases
        ptr_end, old_ptr = self._write_batch_common(batch_size)

        if ptr_end <= self._capacity:
            # No wrap-around writing
            return self._write_no_wraparound(batch_data, old_ptr, ptr_end)
        else:
            # Wrap-around writing
            return self._write_wraparound(batch_data, batch_size, old_ptr)

    def sample(self, batch_size: int, latest: bool = False) -> dict[str, Tensor]:
        """Sample a random batch from the buffer.

        Args:
            batch_size: Number of samples to return

        Returns:
            dictionary containing sampled torch tensors
        """
        assert self._size >= batch_size > 0, f"Buffer size {self._size}, but required {batch_size}"
        if not latest:
            idxs = torch.randint(
                0, self._size, (batch_size,), dtype=torch.int64, device=self._device
            )
        else:
            # generate continuous indices in [ptr-batch_size, ptr) and map to [0, self._size)
            idxs = torch.arange(
                self._ptr - batch_size, self._ptr, dtype=torch.int64, device=self._device
            )
            idxs = idxs % self._size

        return self.sample_by_idxs(idxs)

    def sample_by_idxs(self, idxs: NDArray[np.int64] | Tensor) -> dict[str, Tensor]:
        """Sample data by specific indices.

        Args:
            idxs: Numpy array or torch tensor of indices to sample

        Returns:
            dictionary containing sampled torch tensors
        """
        assert self._initialized, "Memory not initialized. Call add_batch first."

        # Convert numpy array to torch tensor if necessary
        if isinstance(idxs, np.ndarray):
            idxs = torch.from_numpy(idxs)

        # Check if all indices are within valid range
        if torch.any((idxs >= self._size) | (idxs < 0)):
            raise ValueError(
                f"Invalid indices: indices must be in range [0, {self._size}), got {idxs}"
            )

        idxs = idxs.to(self._device)

        return {key: arr[idxs] for key, arr in self._data.items()}

    def _write_no_wraparound(self, batch_data: dict[str, Tensor], start: int, end: int) -> Tensor:
        """Write data without wrap-around."""
        for key, data in batch_data.items():
            self._data[key][start:end] = data

        return torch.arange(start, end, dtype=torch.int64)

    def _write_wraparound(self, batch_data: dict[str, Tensor], batch_size: int, ptr: int) -> Tensor:
        """Write data with wrap-around."""
        head_size = self._capacity - ptr
        tail_size = batch_size - head_size

        # Write data to each field
        for key, data in batch_data.items():
            self._data[key][ptr:] = data[:head_size]
            self._data[key][:tail_size] = data[head_size:]

        # Return written indices
        head_indices = torch.arange(ptr, self._capacity, dtype=torch.int64)
        tail_indices = torch.arange(0, tail_size, dtype=torch.int64)
        return torch.cat([head_indices, tail_indices])

    def _init_memory_if_necessary(self, **batch_data: Tensor) -> None:
        """Initialize memory allocation based on the first batch of data.

        Args:
            **batch_data: Keyword arguments containing torch tensors
        """
        if self._initialized:
            return

        # Pre-allocate memory for each field
        for key, data in batch_data.items():
            shape = (self._capacity, *data.shape[1:])
            self._device = data.device
            self._data[key] = torch.empty(shape, dtype=data.dtype, device=data.device)

        self._initialized = True
