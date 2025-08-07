import matplotlib.pyplot as plt
import numpy as np

from practice.utils_for_coding.scheduler_utils import ExponentialSchedule


def plot_exponential_schedule() -> None:
    """Plot ExponentialSchedule with different parameter configurations."""

    # Time steps to evaluate
    time_steps = np.arange(0, 1000, 1)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("ExponentialSchedule Visualization", fontsize=16)

    # Configuration 1: Different decay rates
    ax1 = axes[0, 0]
    decay_rates = [0.001, 0.005, 0.01, 0.02]
    colors = ["blue", "green", "red", "orange"]

    for decay_rate, color in zip(decay_rates, colors):
        schedule = ExponentialSchedule(start_e=1.0, end_e=0.1, decay_rate=decay_rate)
        values = [schedule(int(t)) for t in time_steps]
        ax1.plot(time_steps, values, color=color, label=f"decay_rate={decay_rate}")

    ax1.set_title("Effect of Different Decay Rates")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Epsilon Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Configuration 2: Different start values
    ax2 = axes[0, 1]
    start_values = [0.8, 1.0, 1.2, 1.5]

    for start_e, color in zip(start_values, colors):
        schedule = ExponentialSchedule(start_e=start_e, end_e=0.1, decay_rate=0.005)
        values = [schedule(int(t)) for t in time_steps]
        ax2.plot(time_steps, values, color=color, label=f"start_e={start_e}")

    ax2.set_title("Effect of Different Start Values")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Epsilon Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Configuration 3: Different end values
    ax3 = axes[1, 0]
    end_values = [0.01, 0.05, 0.1, 0.2]

    for end_e, color in zip(end_values, colors):
        schedule = ExponentialSchedule(start_e=1.0, end_e=end_e, decay_rate=0.005)
        values = [schedule(int(t)) for t in time_steps]
        ax3.plot(time_steps, values, color=color, label=f"end_e={end_e}")

    ax3.set_title("Effect of Different End Values")
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("Epsilon Value")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Configuration 4: Q-learning specific configurations
    ax4 = axes[1, 1]

    # Configuration from taxi config
    taxi_schedule = ExponentialSchedule(start_e=0.05, end_e=0.01, decay_rate=-0.0005)
    taxi_values = [taxi_schedule(int(t)) for t in time_steps]
    ax4.plot(
        time_steps, taxi_values, color="purple", label="Taxi Config (negative decay)", linewidth=2
    )

    # Typical epsilon-greedy configuration
    typical_schedule = ExponentialSchedule(start_e=1.0, end_e=0.05, decay_rate=0.001)
    typical_values = [typical_schedule(int(t)) for t in time_steps]
    ax4.plot(time_steps, typical_values, color="brown", label="Typical ε-greedy", linewidth=2)

    # Fast decay configuration
    fast_schedule = ExponentialSchedule(start_e=1.0, end_e=0.1, decay_rate=0.01)
    fast_values = [fast_schedule(int(t)) for t in time_steps]
    ax4.plot(time_steps, fast_values, color="darkgreen", label="Fast decay", linewidth=2)

    ax4.set_title("Q-learning Specific Configurations")
    ax4.set_xlabel("Time Steps")
    ax4.set_ylabel("Epsilon Value")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_schedule_comparison() -> None:
    """Compare ExponentialSchedule with other schedule types."""

    # Import other schedule types
    from practice.utils_for_coding.scheduler_utils import ConstantSchedule, LinearSchedule

    time_steps = np.arange(0, 500, 1)

    plt.figure(figsize=(12, 8))

    # Exponential schedule
    exp_schedule = ExponentialSchedule(start_e=1.0, end_e=0.1, decay_rate=0.005)
    exp_values = [exp_schedule(int(t)) for t in time_steps]
    plt.plot(time_steps, exp_values, "b-", label="Exponential Schedule", linewidth=2)

    # Linear schedule
    linear_schedule = LinearSchedule(start_e=1.0, end_e=0.1, duration=400)
    linear_values = [linear_schedule(int(t)) for t in time_steps]
    plt.plot(time_steps, linear_values, "r--", label="Linear Schedule", linewidth=2)

    # Constant schedule
    const_schedule = ConstantSchedule(value=0.5)
    const_values = [const_schedule(int(t)) for t in time_steps]
    plt.plot(time_steps, const_values, "g:", label="Constant Schedule", linewidth=2)

    plt.title("Schedule Type Comparison", fontsize=16)
    plt.xlabel("Time Steps")
    plt.ylabel("Epsilon Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_negative_decay_analysis() -> None:
    """Analyze the behavior of negative decay rates as used in the taxi config."""

    time_steps = np.arange(0, 2000, 1)

    plt.figure(figsize=(12, 6))

    # Negative decay rate (as in taxi config)
    neg_schedule = ExponentialSchedule(start_e=0.05, end_e=0.01, decay_rate=-0.0005)
    neg_values = [neg_schedule(int(t)) for t in time_steps]
    plt.plot(time_steps, neg_values, "r-", label="Negative decay (-0.0005)", linewidth=2)

    # Positive decay rate (typical usage)
    pos_schedule = ExponentialSchedule(start_e=0.05, end_e=0.01, decay_rate=0.0005)
    pos_values = [pos_schedule(int(t)) for t in time_steps]
    plt.plot(time_steps, pos_values, "b-", label="Positive decay (+0.0005)", linewidth=2)

    # Show the mathematical difference
    plt.axhline(y=0.05, color="gray", linestyle="--", alpha=0.5, label="start_e = 0.05")
    plt.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5, label="end_e = 0.01")

    plt.title("Negative vs Positive Decay Rate Analysis", fontsize=16)
    plt.xlabel("Time Steps")
    plt.ylabel("Epsilon Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.06)

    # Add text annotation explaining the behavior
    plt.text(
        1000,
        0.045,
        "Negative decay:\nIncreases from start_e to end_e",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )
    plt.text(
        1000,
        0.025,
        "Positive decay:\nDecreases from start_e to end_e",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    plt.show()


if __name__ == "__main__":
    print("Plotting ExponentialSchedule visualizations...")

    # Plot main schedule variations
    plot_exponential_schedule()

    # Compare with other schedule types
    plot_schedule_comparison()

    # Analyze negative decay behavior
    plot_negative_decay_analysis()

    print("All plots displayed!")
