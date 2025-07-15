# Enhanced REINFORCE with Curiosity-Driven Exploration

This exercise implements an enhanced version of REINFORCE with curiosity-driven exploration using RND (Random Network Distillation) and reward shaping techniques.

## Files
- `enhanced_reinforce.py`: Main enhanced REINFORCE implementation containing:
  - `EnhancedReinforceConfig`: Configuration dataclass for the enhanced algorithm
  - `ReinforceContext`: Context containing environment, policy, and rewarders
  - `EpisodeBuffer`: Buffer for storing episode data with intrinsic rewards
  - `EnhancedReinforceTrainer`: Main trainer class with baseline and entropy regularization

- `curiosity_exercise.py`: RND (Random Network Distillation) `EXERCISE` containing:
  - `RNDNetwork1D`: Neural network for RND-based intrinsic motivation
  - `RNDReward`: Reward class implementing RND curiosity mechanism
  - `RND1DNetworkConfig`: Configuration for RND reward system

- `config_mountain_car.py`: Configuration file for MountainCar-v0 environment
- `tests/test_curiosity_train.py`: Comprehensive test suite

## Command
Train
```bash
# Train the model on MountainCar-v0 with RND curiosity
python practice/cli.py --config practice/exercise4_curiosity/config_mountain_car.py
```

Play with trained model and generate video
```bash
python practice/cli.py --config practice/exercise4_curiosity/config_mountain_car.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise4_curiosity/config_mountain_car.py --push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise4_curiosity/config_mountain_car.py --push_to_hub --username myuser --skip_play
```

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise4_curiosity/tests/ -v
```

## Architecture

Algorithm Features
- **Enhanced REINFORCE**: Policy gradient with optional baseline and entropy regularization
- **RND Curiosity**: Random Network Distillation for intrinsic motivation
- **Reward Shaping**: X-Direction reward shaping for improved exploration
- **Gradient Accumulation**: Support for accumulating gradients over multiple episodes
- **Baseline Reduction**: Exponential moving average baseline to reduce variance

Training Flow
1. **Environment Setup**: Initialize environment, policy network, and rewarders
2. **Episode Collection**: Run episodes and collect trajectories
3. **Intrinsic Reward Calculation**: Compute RND-based curiosity rewards
4. **Return Calculation**: Compute discounted returns with combined rewards
5. **Baseline Update**: Update baseline using exponential moving average
6. **Policy Update**: Compute policy gradient with advantage estimation
7. **Logging**: Record training metrics and rewarder contributions

Key Components
- **EnhancedReinforceTrainer**: Orchestrates the training process with multiple reward sources
- **RNDReward**: Implements curiosity-driven exploration using network distillation
- **EpisodeBuffer**: Stores episode data with support for multiple reward types
- **ReinforceContext**: Contains all training components including rewarders
- **Reinforce1DNet**: Policy network (from exercise3 implementation)

## Environments

MountainCar-v0
- **Observation Space**: 2D (position, velocity)
- **Action Space**: 3 discrete actions (left, no-op, right)
- **Goal**: Reach the flag on the right side of the mountain
- **Configuration**: `config_mountain_car.py`
- **Reward Components**:
  - Extrinsic: Sparse reward from environment
  - RND Curiosity: Intrinsic reward for novel states
  - X-Direction Shaping: Reward for moving toward goal

## RND (Random Network Distillation)

The curiosity mechanism works by:
1. **Target Network**: Randomly initialized network (frozen)
2. **Predictor Network**: Trainable network that tries to predict target outputs
3. **Intrinsic Reward**: L2 norm of prediction error
4. **Normalization**: Running mean/variance normalization for stability
5. **Scheduling**: Beta parameter controls curiosity strength over time

## Environment Support

Currently supports 1D observation environments. The implementation includes:
- Modular reward system supporting multiple reward sources
- Configurable reward scheduling and normalization
- Extensible to different observation and action spaces

## Future Enhancements

Potential improvements:
- Integration with other curiosity methods (ICM, NGU)
- Multi-environment training
- Distributed training support
- Advanced baseline methods (value function approximation)
- Support for continuous action spaces
