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
