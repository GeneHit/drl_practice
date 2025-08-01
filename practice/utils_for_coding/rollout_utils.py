import torch
from torch import Tensor


def filter_data(target: Tensor, mask: Tensor) -> Tensor:
    """Filter the data.

    Parameters
    ----------
        target: Tensor, shape [T, N]
        mask: Tensor, shape [T, N]

    Returns
    -------
    The valid data of the rollout (d (<= T * N) is the valid data length):
        targets: Tensor, shape [d, ]
    """
    n_dim = mask.ndim
    # filter out the invalid data: [T, N] -> [d, ]
    valid_mask = mask.view(-1)
    if target.ndim > n_dim:
        return target.view(-1, *target.shape[n_dim:])[valid_mask]
    return target.view(-1)[valid_mask]


def get_good_transition_mask(dones: Tensor, pre_last_dones: Tensor) -> Tensor:
    """Get the good transition mask.

    Because of the gymnasium's AutoReset wrapper, the rollout data may be discontinuous, which means
    the rollout contains multiple episodes's data.
    Bad transition between two episodes, for one environment:
        - if it's done in n-step, the next_states of n+1 step is the first of the next episode
        - so, transition (s_n, a_n, r_n, s_n+1) is a bad data. Have to be ignored.

    Example:
        1. s_n-1, a_n-1, r_n-1, s_n     (last step of episode 1)
        2. s_n, a_n, r_n, s_n+1         (bad transition between two episodes)
        3. s_n+1, a_n+1, r_n+1, s_n+2   (first step of episode 2)

    Have to throw away the data of the bad transition between two episodes, don't use its data
    and compute graph of backward.

    Parameters
    ----------
        dones: Tensor, shape [T, N]
        pre_last_dones: Tensor, shape [N, ]
            The last done of the previous rollout.

    Returns
    -------
        mask: Tensor, shape [T, N]
            The good transition mask.
    """
    # filter the bad transition between two episodes: pre_dones == 1
    prev_dones = torch.empty_like(dones, dtype=torch.bool)
    prev_dones[0] = pre_last_dones
    prev_dones[1:] = dones[:-1]
    # return the mask and the last done of the current rollout
    return ~prev_dones
