from typing import Any

import numpy as np
from numpy.typing import NDArray


class SumTree:
    """SumTree for prioritized experience replay.

    SumTree is a binary tree where the parent node is the sum of the two child nodes. It makes
    sampling more efficient.
    - The leaf nodes are the priorities of the experiences.
    - The root node is the total priority of all experiences.

    Reference:
    - https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 1:
            raise ValueError("capacity must be greater than 1")
        self._capacity = int(capacity)
        self._tree = np.zeros(2 * self._capacity - 1, dtype=np.float64)
        self._used = np.zeros(self._capacity, dtype=np.bool_)
        self._leaf_start = self._capacity - 1

    @property
    def total_priority(self) -> float:
        return float(self._tree[0])

    def __len__(self) -> int:
        return int(self._used.sum())

    def update(self, idxs: NDArray[np.integer[Any]], priorities: NDArray[np.float32]) -> None:
        idxs = np.asarray(idxs, dtype=np.int64)
        ps = np.asarray(priorities, dtype=np.float32)

        if idxs.ndim != 1 or ps.ndim != 1 or len(idxs) != len(ps):
            raise ValueError("idxs/priorities must be 1-D and of equal length")
        if (idxs < 0).any() or (idxs >= self._capacity).any():
            raise ValueError("index out of range")
        if not np.isfinite(ps).all():
            raise ValueError("priorities must be finite")
        if (ps < -1e-8).any():
            raise ValueError("priorities must be non-negative")

        # clean up invalid values
        # ps = np.nan_to_num(ps, nan=0.0, posinf=0.0, neginf=0.0)
        ps = np.maximum(ps, 0.0)

        self._used[idxs] = True  # mark as used

        for i, p in zip(idxs, ps.astype(np.float64)):
            leaf = self._leaf_start + int(i)
            delta = p - self._tree[leaf]
            if abs(delta) < 1e-6:
                continue

            self._tree[leaf] = p
            # accumulate the difference along the parent chain
            parent = (leaf - 1) // 2
            while True:
                self._tree[parent] += delta
                if not np.isfinite(self._tree[parent]) or (self._tree[parent] < -1e-8):
                    raise ValueError("The tree get invalid values.")
                if parent == 0:
                    break
                parent = (parent - 1) // 2

    def get_leaves(
        self, values: NDArray[np.float32 | np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Get the leaves of the tree given the values.

        Args:
            values: The values to retrieve.

        Returns:
            tuple[NDArray[np.int64], NDArray[np.float32]]: The leaves of the tree.
                The first element is the indices of the leaves.
                The second element is the priorities of the leaves.
        """
        size = len(self)
        k = len(values)
        if not (0 < k <= size):
            raise ValueError(f"buffer size={size}, but requested {k} samples")
        tot = self.total_priority
        if not np.isfinite(tot) or (tot < -1e-8):
            raise ValueError("total priority must be positive and finite")

        data_idx = np.empty(k, dtype=np.int64)
        prios = np.empty(k, dtype=np.float32)
        for j, val in enumerate(values):
            leaf = self._retrieve(0, float(val))
            data_idx[j] = leaf - self._leaf_start
            prios[j] = self._tree[leaf].astype(np.float32)
        return data_idx, prios

    def sample(self, k: int) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Sample k leaves from the tree with segment sampling.

        Args:
            k: The number of leaves to sample.

        Returns:
            tuple[NDArray[np.int64], NDArray[np.float32]]: The leaves of the tree.
                The first element is the indices of the leaves.
                The second element is the priorities of the leaves.
        """
        tot = self.total_priority
        segment = tot / float(k)

        # segment sampling
        bases = np.arange(k, dtype=np.float64) * segment
        v = bases + np.random.uniform(0.0, segment, size=k).astype(np.float64)

        return self.get_leaves(v)

    def tree_metrics(self) -> dict[str, float]:
        """Get the metrics of the tree."""
        leaf_values = self._tree[self._leaf_start :][self._used]
        leaves_sum = leaf_values.sum()
        root = self.total_priority
        ESS = 1 / (np.sum(np.power(leaf_values, 2)) + 1e-8)
        top5pct_idx = max(1, int(0.05 * len(leaf_values)))
        top5pct_mass = np.sort(leaf_values)[-top5pct_idx:].sum() / (leaves_sum + 1e-8)
        return {
            # tree consistency: not ≈0 means there is a bug in the implementation or update
            "per/root_minus_leafsum": abs(root - leaves_sum),
            # whether the buffer is dominated by a few samples: 0.0 means no, 1.0 means yes
            "per/top5pct_mass": top5pct_mass,
            # effective sample size: larger is more diverse
            "per/ESS": ESS,
        }

    def _retrieve(self, node: int, s: float) -> int:
        """Retrieve the leaf node index given the value s.

        Args:
            node: The current node index.
            s: The value to retrieve.

        Returns:
            int: The root or leaf node index.
                If the recursive arrives the leaf node (last level), it is the leaf node index.
                Otherwise, it is the root node index.
        """
        left = 2 * node + 1
        if left >= self._tree.size:
            return node  # return the leaf node index

        left_sum = float(self._tree[left])
        if s <= left_sum:
            return self._retrieve(left, s)  # go left
        else:
            return self._retrieve(left + 1, s - left_sum)  # go right
