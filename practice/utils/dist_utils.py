import os
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn


def get_device(target: str | None = None) -> torch.device:
    """Get the device.

    Args:
        target: The target device.
            - "cpu": CPU
            - "cuda": GPU
            - "mps": Apple Silicon GPU
            - None: Auto-detect the best device

    Returns:
        The device.
    """
    if target is not None:
        if target == "cuda":
            return get_cuda_dist_device()
        return torch.device(target)

    if torch.cuda.is_available():
        return get_cuda_dist_device()

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def get_cuda_dist_device() -> torch.device:
    """Get the device for the distributed environment."""
    assert torch.cuda.is_available(), "CUDA is not available"
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cuda")


def auto_init_distributed(device: torch.device, model: nn.Module) -> nn.Module:
    """
    Initialize the distributed environment and move the model to the correct device.

    Args:
        device: The device to use for the distributed environment.
        model: The model to move to the correct device.

    Returns:
        nn.Module: The model with the correct device. When the distributed environment is not
                    initialized, the model is not wrapped by DistributedDataParallel.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1:
        print("[distributed] Not running distributed, will run single process.")
        return model.to(device)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    assert device.type in ("cuda", "cpu"), "Only support distributed on cuda and cpu"
    backend = "nccl" if device.type == "cuda" else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        assert device.index == local_rank, f"device.index={device.index} != local_rank={local_rank}"

    model.to(device)
    if dist.is_initialized() and device.type == "cuda":
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif dist.is_initialized() and device.type == "cpu":
        model = nn.parallel.DistributedDataParallel(model)

    print(f"[distributed] initialized: rank={rank} world_size={world_size} backend={backend}")
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return cast(nn.Module, model.module)
    return model


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    # single process
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


def is_distributed() -> bool:
    """Check if the distributed environment is initialized.

    Returns:
        Whether the distributed environment is initialized.
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get the world size."""
    return int(os.environ.get("WORLD_SIZE", 1))
