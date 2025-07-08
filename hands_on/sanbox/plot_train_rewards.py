import json

import matplotlib.pyplot as plt


def main() -> None:
    file = "results/exercise1_q/frozen_lake/train_result.json"

    with open(file, "r") as f:
        data = json.load(f)

    episode_rewards = data["episode_rewards"]

    plt.plot(episode_rewards)
    plt.show()


if __name__ == "__main__":
    main()
