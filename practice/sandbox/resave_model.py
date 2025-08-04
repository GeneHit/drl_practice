import torch
import torch.nn as nn


def main() -> None:
    path = "results/exercise9_sac/pendulum_used/"
    name = "sac_pendulum.pth"
    device = torch.device("cpu")
    net: nn.Module = torch.load(path + name, map_location=device, weights_only=False)

    state_dict_name = "state_dict.pt"
    torch.save(net.state_dict(), path + state_dict_name)


if __name__ == "__main__":
    main()
