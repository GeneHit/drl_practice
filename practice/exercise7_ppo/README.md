# PPO

This exercise implements a Proximal Policy Optimization (PPO).


## Files
- `ppo_exercise.py`: PPO `EXERCISE`.
- `config_mountain_car.py`: Configuration file for MountainCar-v1 environment
- `config_lunar_1d.py`: Configuration file for LunarLander-v3 environment

Different games:
1. `MountainCar`, a easy game is used for verifying the implementation
2. `LunarLander`, a harder game, you should train it to have a good score.


## Command
Train
```bash
# Train the model on CartPole-v1
python practice/cli.py --config practice/exercise7_ppo/config_cartpole.py
```

Play with trained model and generate video
```bash
python practice/cli.py --config practice/exercise7_ppo/config_cartpole.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise7_ppo/config_cartpole.py --push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise7_ppo/config_cartpole.py --push_to_hub --username myuser --skip_play
```

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise7_ppo/tests/ -v
```

## Parameter Tuning
### CartPole
