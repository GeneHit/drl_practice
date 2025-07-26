# drl_practice
Practice the Deep Reinforcement Learning (DRL) with the [gymnasium](https://gymnasium.farama.org/).
- Easy hands-on on our laptop (like Mac/window/linux).
- No Atari environments (because of Mac).
- No long-time training.

## How to practice
Check the [Command Guide](./practice/README.md) doc, which provides the
step-by-step commands for installing and running every exercise.
The things you need to do:
1. Create the python env
2. For a exercise, implement the function for every `NotImplementedError` in the `*_exercise.py` file .
3. then train it with the provided command.
4. [Optional] generate the video and push the video/result to the HuggingFace.


## Exercises
Don't choose too hard game and big neural network. But you can try it by yourself.

| Exercise | Algorithm | Verification Game | For Challenge | State | Action |
|----------|-----------|-------------------|---------------|-------|--------|
| [1. q_learning](./practice/exercise1_q/README.md) | Q Table | FrozenLake | Taxi | D | D |
| [2. dqn](./practice/exercise2_dqn/README.md) | Deep Q Network | 1D LunarLander-v3 | img LunarLander-v3 | C | D |
| [3. reinforce](./practice/exercise3_reinforce/README.md) | Reinforce (Monte Carlo) | CartPole-v1 | MountainCar-v0 | C | D |
| [4. curiosity](./practice/exercise4_curiosity/README.md) | Curiosity (with Reinforce, baseline, shaping reward) | | MountainCar-v0 | C | D |
| [5. A2C](./practice/exercise5_a2c/README.md) | A2C + GAE or A2C + TD-n | CartPole-v1 | LunarLander-v3 | C | D |
| [6. A3C](./practice/exercise6_a3c/README.md) | A3C (using A2C+GAE) | CartPole-v1 | LunarLander-v3 | C | D |
| [7. PPO](./practice/exercise7_ppo/README.md) | PPO | CartPole-v1 | LunarLander-v3 | C | D |
| [8. TD3](./practice/exercise8_td3/README.md) | Twin Delayed DDPG (TD3) | Pendulum-v1 | Walker2d-v5 | C | C |
| [9. SAC](./practice/exercise9_sac/README.md) | | Pendulum-v1 | Walker2d-v5 | C | C |

where, C: continuous; D: discrete


## Motivation
After studying
the HuggingFace's [DRL course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) and
Pieter Abbeel's [The Foundations of Deep RL in 6 Lectures](https://www.youtube.com/watch?v=2GwBez0D20A&list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0),
I want to have a deeper and broader understanding through the coding.


## Other

1. [RL Algorithms](./practice/infos/rl_algorithm.md)
2. OpenAI's [Spining Up](https://spinningup.openai.com/en/latest/)
3. [Stable Baseline3](https://stable-baselines3.readthedocs.io/en/master/index.html)
