import gymnasium as gym

from common.evaluation_utils import play_game_once
from exercise1_q_learning.q_learning_train import QTable


def main() -> None:
    # load the q_table from the file
    q_table = QTable.load("results/q_table.pkl")

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
        render_mode="rgb_array",
    )

    # play the game
    play_game_once(
        env=env,
        policy=q_table,
        show_image=True,
        save_video=True,
        video_pathname="results/q_learning_video.mp4",
    )


if __name__ == "__main__":
    main()
