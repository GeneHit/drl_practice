# drl_practice
Practice the Deep Reinforcement Learning (DRL) with the gymnasium.

## How to practice
Check the [Command Guide](./hands_on/README.md) doc, which provides the
step-by-step commands for installing and running every exercise.
The things you need to do:
1. Create the python env
2. For a exercise, implement the function for every `NotImplementedError` in the `*_exercise.py` file .
3. then train it.
4. [Optional] generate the video and push the video/result to the HuggingFace.


## Exercises

| Exercise | Algorithm | Easy Game | Hard Game |
|----------|-----------|-----------|-----------|
| [1. q_learning](./practice/exercise1_q/README.md) | Q Table | FrozenLake | Taxi |
| [2. dqn](./practice/exercise2_dqn/README.md) | Deep Q Network | 1D LunarLander-v3 | img LunarLander-v3 |
| [3. reinforce](./practice/exercise3_reinforce/README.md) | Reinforce (Monte Carlo) | CartPole-v1 | MountainCar-v0 |
| [4. curiosity](./practice/exercise4_curiosity/README.md) | Curiosity (with enhanced Reinforce) | | MountainCar-v0 |


## Motivation
After studying the HuggingFace's
[DRL course](https://huggingface.co/learn/deep-rl-course/unit0/introduction),
I want to have a better understanding through the coding. So, some code
is came from or bassed on the course's code. Thanks for this greate course.
