import json

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # file = "results/exercise1_q/frozen_lake/train_result.json"
    mean_num = 100
    file = "results/exercise2_dqn/lunar_2d_1/train_result.json"

    with open(file, "r") as f:
        data = json.load(f)

    episode_rewards = data["episode_rewards"]

    # Calculate rolling mean over mean_num episodes
    if len(episode_rewards) >= mean_num:
        rolling_mean = []
        for i in range(len(episode_rewards) - mean_num + 1):
            mean_reward = np.mean(episode_rewards[i : i + mean_num])
            rolling_mean.append(mean_reward)

        # Plot both raw rewards and rolling mean
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, alpha=0.3, label="Raw rewards")
        plt.title("Raw Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(
            range(mean_num - 1, len(episode_rewards)),
            rolling_mean,
            label=f"Rolling mean ({mean_num} episodes)",
        )
        plt.title(f"Rolling Mean Rewards ({mean_num} episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print(
            f"Not enough episodes for rolling mean. Need at least {mean_num} episodes, got {len(episode_rewards)}"
        )
        plt.plot(episode_rewards)
        plt.title("Episode Rewards (insufficient data for rolling mean)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()


if __name__ == "__main__":
    main()
