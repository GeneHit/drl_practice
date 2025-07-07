import gymnasium as gym


def main() -> None:
    # First, create the environment
    # env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    env = gym.make("LunarLander-v3")

    # obs = env.observation_space.sample()
    # print("Sample observation", obs)
    # print(f"type(obs): {type(obs)}")
    # print(f"obs.dtype: {obs.dtype}")
    # print(f"obs.shape: {obs.shape}")

    # act = env.action_space.sample()
    # print(f"Action Space Shape: {env.action_space.n}")
    # print("Sample action", act)
    # print(f"type(act): {type(act)}")
    # print(f"act.dtype: {act.dtype}")
    # return

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
