# Model-based Policy Optimization (MBPO) (todo)

Paper: https://arxiv.org/abs/1906.08253
MBPO first learns a model of the environment dynamics, then uses this model to generate additional
training data.

How Does It Work?
1. **Learn the Environment Model**: MBPO starts by learning a neural network that predicts
the next state and reward given the current state and action.
2. **Generate Synthetic Data**: using the learned model.
3. **Train the Policy**: using both real experience and the synthetic data from the model.

## Files
## Commands
Train the model (change `reacher` to `pusher` if needed)
```bash
python practice/cli.py --config practice/exercise12_mbpo/config_reacher.py
```

Play with trained model and generate video
```bash
python practice/cli.py --config practice/exercise12_mbpo/config_ddp_reacher.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise12_mbpo/config_ddp_reacher.py --mode push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise12_mbpo/config_ddp_reacher.py --mode push_to_hub --username myuser --skip_play
```

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise12_mbpo/tests/ -v
```

## Parameter Tuning


## More for MBPO

Key Benefits
- **Sample Efficiency**: Requires fewer real environment interactions to achieve good performance
- **Data Augmentation**: Generates additional training data without extra environment steps
- **Stability**: Often more stable than pure model-free methods
- **Performance**: Can achieve better final performance in many environments

MBPO is particularly useful when:
- Environment interactions are expensive or time-consuming
- You have limited real-world data
- You need to learn complex policies efficiently
- The environment dynamics are relatively smooth and predictable
