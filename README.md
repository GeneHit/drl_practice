# drl_practice
Practice the Deep Reinforcement Learning (DRL) with the [gymnasium](https://gymnasium.farama.org/).
- Easy hands-on on our laptop (like Mac/window/linux).
- No long-time training.

## How to practice
Check the [Command Guide](./practice/README.md) for the step-by-step commands:
- Create the conda env with pip.
- Exercise
    1. For a exercise, implement all `NotImplementedError`s in the `*_exercise.py` file .
    2. then train it with the provided command.
    3. [Optional] generate the video and push the video/result to the HuggingFace.


## Exercises
Don't choose too hard game and big neural network. But you can try it by yourself.

|            Exercise                                       |         Algorithm         | Verification Game |      For Challenge    | State | Action |
|-----------------------------------------------------------|---------------------------|-------------------|-----------------------|-------|--------|
| [1. q_learning](./practice/exercise1_q/README.md)         | Q Table                   | FrozenLake        | Taxi                  | 📊    | 📊 |
| [2. dqn](./practice/exercise2_dqn/README.md)              | Deep Q Network -> Rainbow | 1D LunarLander-v3 | img LunarLander-v3    | 🌊    | 📊 |
| [3. reinforce](./practice/exercise3_reinforce/README.md)  | Reinforce (Monte Carlo)   | CartPole-v1       | -                     | 🌊    | 📊 |
| [4. curiosity](./practice/exercise4_curiosity/README.md)  | Curiosity (Reinforce, baseline, shaping reward)| - | MountainCar-v0   | 🌊    | 📊 |
| [5. A2C](./practice/exercise5_a2c/README.md)              | A2C+GAE (or A2C+TD-n)     | CartPole-v1       | LunarLander-v3        | 🌊    | 📊 |
| [6. A3C](./practice/exercise6_a3c/README.md)              | A3C (using A2C+GAE)       | CartPole-v1       | LunarLander-v3        | 🌊    | 📊 |
| [7. PPO](./practice/exercise7_ppo/README.md)              | PPO                       | CartPole-v1       | LunarLander-v3        | 🌊    | 📊 |
| [8. TD3](./practice/exercise8_td3/README.md)              | Twin Delayed DDPG (TD3)   | Pendulum-v1       | Walker2d-v5           | 🌊    | 🌊 |
| [9. SAC](./practice/exercise9_sac/README.md)              | SAC (Soft Actor-Critic)   | Pendulum-v1       | Walker2d-v5           | 🌊    | 🌊 |
| [10. PPO+DDP](./practice/exercise10_ddp_ppo/README.md)    | PPO+Curiosity             | Reacher-v5        | Pusher-v5             | 🌊    | 🌊 |
| [11. SAC+DDP](./practice/exercise11_ddp_sac/README.md)    | SAC+PER                   | Reacher-v5        | Pusher-v5             | 🌊    | 🌊 |
| [12. MBPO](./practice/exercise12_mbpo/README.md)          | Model-based Policy Optim. | Pusher-v5         | Walker2d-v5           | 🌊    | 🌊 |
where, 🌊: Continuous,  📊: Discrete


## Motivation
After studying
the HuggingFace's [DRL course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) and
Pieter Abbeel's [The Foundations of Deep RL in 6 Lectures](https://www.youtube.com/watch?v=2GwBez0D20A&list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0),
I want to have a deeper and broader understanding through the coding.


## Other

1. [RL Algorithms](./practice/infos/rl_algorithm.md)
2. OpenAI's [Spining Up](https://spinningup.openai.com/en/latest/)
3. [Stable Baseline3](https://stable-baselines3.readthedocs.io/en/master/index.html)
