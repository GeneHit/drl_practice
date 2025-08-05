import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def filter_and_save_scalars(
    input_event_file: str, output_dir: str, step_interval: int, ignore_tags: list[str]
) -> None:
    # === read the original log ===
    print(f"reading {input_event_file}...")
    ea = EventAccumulator(input_event_file)
    ea.Reload()

    # === create a new writer ===
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir)

    # === filter the log ===
    print(f"{ea.Tags()['scalars']=}")
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        for event in events:
            if tag in ignore_tags:
                writer.add_scalar(tag, event.value, event.step)
                continue
            if event.step % step_interval == 0:
                writer.add_scalar(tag, event.value, event.step)

    writer.close()


def main() -> None:
    event_dir = "results/exercise8_td3/walker_used/tensorboard"
    output_dir = "results/exercise8_td3/walker_used/tensorboard/reduced"
    step_interval = 10
    ignore_tags = ["episode_reward", "episode_length"]

    # === find all tfevents files ===
    event_files = [
        os.path.join(event_dir, event_file)
        for event_file in os.listdir(event_dir)
        if "tfevents" in event_file
    ]
    event_files.sort()

    for event_file in event_files:
        filter_and_save_scalars(event_file, output_dir, step_interval, ignore_tags)

    print(f"✅ saved filtered TensorBoard log to: {output_dir}")


if __name__ == "__main__":
    main()
