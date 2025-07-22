# drl_practice
Practice the Deep Reinforcement Learning (DRL) with the [gymnasium](https://gymnasium.farama.org/).
- Easy hands-on on our laptop (like Mac/window/linux).
- No complex Atari and Mojuco environments.

## How to practice
Check the [Command Guide](./practice/README.md) doc, which provides the
step-by-step commands for installing and running every exercise.
The things you need to do:
1. Create the python env
2. For a exercise, implement the function for every `NotImplementedError` in the `*_exercise.py` file .
3. then train it.
4. [Optional] generate the video and push the video/result to the HuggingFace.


## Exercises

| Exercise | Algorithm | Easy Game (for verification) | Hard Game (challenge) |
|----------|-----------|-----------|-----------|
| [1. q_learning](./practice/exercise1_q/README.md) | Q Table | FrozenLake | Taxi |
| [2. dqn](./practice/exercise2_dqn/README.md) | Deep Q Network | 1D LunarLander-v3 | img LunarLander-v3 |
| [3. reinforce](./practice/exercise3_reinforce/README.md) | Reinforce (Monte Carlo) | CartPole-v1 | MountainCar-v0 |
| [4. curiosity](./practice/exercise4_curiosity/README.md) | Curiosity (with Reinforce, baseline, shaping reward) | | MountainCar-v0 |
| [5. A2C](./practice/exercise5_a2c/README.md) | A2C + GAE or A2C + TD-n | CartPole-v1 | LunarLander-v3 |
| [6. A3C](./practice/exercise6_a3c/README.md) | A3C (using A2C+GAE) | CartPole-v1 | LunarLander-v3 |
| [7. PPO](./practice/exercise7_ppo/README.md) | PPO | CartPole-v1 | LunarLander-v3 |
| 8. DDPG | | CartPole-v1 | LunarLander-v3 |
| 9. SAC | | CartPole-v1 | LunarLander-v3 |


## Motivation
After studying the HuggingFace's
[DRL course](https://huggingface.co/learn/deep-rl-course/unit0/introduction),
I want to have a deeper and broader understanding through the coding. Some code and config
are based on the course. Thanks for this greate course.
