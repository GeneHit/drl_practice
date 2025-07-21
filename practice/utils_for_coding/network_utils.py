import torch
import torch.nn as nn


def init_weights(layer: nn.Module) -> None:
    """
    Initialize weights for Conv2d and Linear layers using orthogonal initialization
    with appropriate gain for ReLU activations, and zero biases.
    """
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        # Use orthogonal initialization
        nn.init.orthogonal_(layer.weight, gain=int(nn.init.calculate_gain("relu")))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def load_checkpoint_if_exists(model: nn.Module, checkpoint_pathname: str) -> None:
    """Load a checkpoint into a model.

    Args:
        model: The model to load the checkpoint into.
        checkpoint_path: The path to the checkpoint.
    """
    if not checkpoint_pathname:
        return

    checkpoint = torch.load(checkpoint_pathname, weights_only=False)
    if isinstance(checkpoint, dict):
        # It's a state_dict
        model.load_state_dict(checkpoint)
    else:
        # It's a full model, extract state_dict
        model.load_state_dict(checkpoint.state_dict())
