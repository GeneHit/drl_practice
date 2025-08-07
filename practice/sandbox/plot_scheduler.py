import matplotlib.pyplot as plt
import numpy as np

from practice.utils_for_coding.scheduler_utils import CosineSchedule


def main() -> None:
    # Specify the scheduler
    step = 1000
    schedule = CosineSchedule(start_e=1.0, end_e=0.1, duration=step, decay_factor=1.0)
    label = "Cosine"

    # Generate time steps and values
    steps = np.arange(0, step, 1)
    values = [schedule(int(t)) for t in steps]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, label=label)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Scheduler Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
