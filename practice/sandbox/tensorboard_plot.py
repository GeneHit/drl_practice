import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def plot_scalar() -> None:
    log_dir = "results/exercise8_td3/pendulum/tensorboard"

    # load event file (auto find the first one)
    event_file = [f for f in os.listdir(log_dir) if "tfevents" in f][0]
    ea = EventAccumulator(os.path.join(log_dir, event_file), size_guidance={"scalars": 1_000_000})
    ea.Reload()

    # print scalar tags
    print("Available scalar tags:", ea.Tags()["scalars"])

    # read noise_std
    noise_std_events = ea.Scalars("action/noise_std")
    steps = [e.step for e in noise_std_events]
    values = [e.value for e in noise_std_events]

    # print step range and number of points
    print(f"Loaded {len(steps)} points for action/noise_std")
    print(f"Step range: {steps[0]} -> {steps[-1]}")

    # visualize
    plt.plot(steps, values)
    plt.xlabel("Step")
    plt.ylabel("action/noise_std")
    plt.title("Raw noise_std points from event file")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_scalar()
