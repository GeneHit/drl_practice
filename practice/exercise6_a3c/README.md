# A3C

This exercise implements a Asynchronous Advantage Actor-Critic (A3C).

Paper: [A3C](https://arxiv.org/abs/1602.01783).

## Files
- `a3c_exercise.py`: A3C `EXERCISE`.
- `config_mountain_car.py`: Configuration file for MountainCar-v1 environment
- `config_lunar_1d.py`: Configuration file for LunarLander-v3 environment

Different games:
1. `MountainCar`, a easy game is used for verifying the implementation
2. `LunarLander`, a harder game, you should train it to have a good score.

## Command
Train
```bash
# Train the model on CartPole-v1
python practice/exercise6_a3c/a3c_cli.py --config practice/exercise6_a3c/config_cartpole.py
```

Play with trained model and generate video
```bash
python practice/cli.py --config practice/exercise6_a3c/config_cartpole.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise6_a3c/config_cartpole.py --mode push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise6_a3c/config_cartpole.py --mode push_to_hub --username myuser --skip_play
```

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise6_a3c/tests/ -v
```

## Parameter Tuning
### LunarLander
1. try: (6 worker, 6 VecEnv) -> (6 worker, 1 VecEnv) -> (6 worker, 2 VecEnv)
2. increase the global_step to 300K from 200K.
