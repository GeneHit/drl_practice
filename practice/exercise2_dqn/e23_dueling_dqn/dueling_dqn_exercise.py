import torch
import torch.nn as nn

from practice.utils_for_coding.network_utils import MLP, init_weights


class DuelingDQN1D(nn.Module):
    """Dueling DQN for 1D state space.

    Reference:
    - https://arxiv.org/abs/1511.06581
    """

    def __init__(
        self, state_n: int, action_n: int, hidden_sizes: tuple[int, ...] = (128, 128)
    ) -> None:
        super().__init__()
        # feature stream
        last_hidden_size = hidden_sizes[-1]
        self.feature = MLP(
            input_dim=state_n,
            hidden_sizes=hidden_sizes[:-1],
            output_dim=last_hidden_size,
            activation=nn.ReLU,
            output_activation=None,
        )
        # state value stream
        self.value_head = nn.Sequential(nn.Linear(last_hidden_size, 1))
        # action advantage stream
        self.advantage_head = nn.Sequential(nn.Linear(last_hidden_size, action_n))
        self.advantage_head.apply(init_weights)
        self.value_head.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, state_dim]
        feat = self.feature(x)
        value = self.value_head(feat)  # [batch, 1]
        advantage = self.advantage_head(feat)  # [batch, action_n]
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))  # [batch, action_n]
        assert isinstance(q, torch.Tensor)  # make mypy happy
        return q
