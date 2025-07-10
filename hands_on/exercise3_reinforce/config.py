import dataclasses


@dataclasses.dataclass
class ReinforceConfig:
    """The config for the reinforce algorithm."""

    gamma: float
    grad_acc: int
    lr: float
    episode: int
    max_steps: int
