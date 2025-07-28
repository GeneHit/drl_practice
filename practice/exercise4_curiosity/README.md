# Enhanced REINFORCE with Curiosity-Driven Exploration

This exercise implements an enhanced version of REINFORCE with curiosity-driven exploration using
Random Network Distillation (RND) and reward shaping techniques.

## Files
- `config_mountain_car.py`: Configuration file for MountainCar-v0 environment
- `curiosity_exercise.py`: RND `EXERCISE` containing:
- `enhanced_reinforce.py`: Main enhanced REINFORCE implementation.
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
python practice/cli.py --config practice/exercise4_curiosity/config_mountain_car.py --mode push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise4_curiosity/config_mountain_car.py --mode push_to_hub --username myuser --skip_play
```

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise4_curiosity/tests/ -v
```


## Reward

MountainCar-v0
- **Reward Components**:
  - Extrinsic: Sparse reward from environment
  - RND Curiosity: Intrinsic reward for novel states
  - X-Direction Shaping: Reward for moving toward goal

### RND (Random Network Distillation)

The curiosity mechanism works by:
1. **Target Network**: Randomly initialized network (frozen)
2. **Predictor Network**: Trainable network that tries to predict target outputs
3. **Intrinsic Reward**: L2 norm of prediction error
4. **Normalization**: Running mean/variance normalization for stability
5. **Scheduling**: Beta parameter controls curiosity strength over time


## Parameter Tuning
1. smaller entropy_coef.
2. RND: smaller beta, add clip and normalize
    - ensure reward is in a same level with the environment reward.
    - bad tries: zeor entropy, no x shaping reward, very small RND beta
3. small policy and RND hidden_sizes: -> [32, 32]
4. TODO: policy has 2 optimal point now
    - local point with the car always go left first (left-right-left-right)
    - better point with the car always go right first (right-left-right).
