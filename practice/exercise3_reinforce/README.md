# Vanilla REINFORCE Exercise

REINFORCE is a policy gradient method that directly optimizes the policy by using the gradient of the expected return.

## Files
- `reinforce_exercise.py`: Main `EXERCISE` containing:
- `config_cartpole.py`: Configuration file for CartPole-v1 environment
- `tests/test_reinforce_train.py`: Comprehensive test suite

`Just use cartpole to verify the vanilla reinforce`.

## Command
Train
```bash
# Train the model on CartPole-v1
python practice/cli.py --config practice/exercise3_reinforce/config_cartpole.py
```

Play with trained model and generate video
```bash
python practice/cli.py --config practice/exercise3_reinforce/config_cartpole.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise3_reinforce/config_cartpole.py --mode push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise3_reinforce/config_cartpole.py --mode push_to_hub --username myuser --skip_play
```
**Replace `myuser` with your HuggingFace account name.**

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise3_reinforce/tests/ -v
```

## Architecture

Algorithm Features:
- Vanilla REINFORCE, Entropy Regularization, Gradient Accumulation

Training Flow
1. **Environment Setup**: Initialize environment and policy network
2. **Episode Collection**: Run episodes and collect trajectories
3. **Return Calculation**: Compute discounted returns for each step
4. **Policy Update**: Compute policy gradient and update parameters
5. **Logging**: Record training metrics and performance


## Future Enhancements

Potential improvements:
- Actor-Critic variants (with baseline)
- Natural policy gradients
- Proximal Policy Optimization (PPO)
- Multi-environment training
- Distributed training support
