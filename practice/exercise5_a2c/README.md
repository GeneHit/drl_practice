# A2C with GAE

This exercise implements a Advantage Actor-Critic (A2C) with [GAE](https://arxiv.org/abs/1506.02438).
Optional: A2C with TD-n.

## Files


## Command
Train
```bash
# Train the model on MountainCar-v0
python practice/cli.py --config practice/exercise5_a2c/config_mountain_car.py
```

Play with trained model and generate video
```bash
python practice/cli.py --config practice/exercise5_a2c/config_mountain_car.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise5_a2c/config_mountain_car.py --push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise5_a2c/config_mountain_car.py --push_to_hub --username myuser --skip_play
```

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise5_a2c/tests/ -v
```
