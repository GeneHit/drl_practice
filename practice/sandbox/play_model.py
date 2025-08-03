import gymnasium as gym
import torch
import torch.nn as nn


def main() -> None:
    env = gym.make("Reacher-v5")
    pathname = "../../results/exercise10_ddp_ppo/reacher_good/model.pth"
    device = torch.device("cpu")
    # load a full model
    net: nn.Module = torch.load(pathname, map_location=device, weights_only=False).to(device)
    net.eval()

    state, _ = env.reset()

    reward_sum = 0.0
    while True:
        action = net.action(state)  # type: ignore
        state, reward, terminated, truncated, _ = env.step(action)
        reward_sum += float(reward)
        if terminated or truncated:
            break

    env.close()
    print(f"Reward sum: {reward_sum:.2f}")


if __name__ == "__main__":
    main()
