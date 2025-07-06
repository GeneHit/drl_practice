import gymnasium as gym


def main() -> None:
    # First, create the environment
    # env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    env = gym.make("LunarLander-v3")

    # Then reset this environment
    env.reset()

    for _ in range(20):
        # Take a random action
        action = env.action_space.sample()
        print("Action taken:", action)

        # Do this action in the environment and get
        # next_state, reward, terminated, truncated and info
        _, _, terminated, truncated, _ = env.step(action)

        # If the game is terminated (in our case we land, crashed) or truncated (timeout)
        if terminated or truncated:
            # Reset the environment
            print("Environment is reset")
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
