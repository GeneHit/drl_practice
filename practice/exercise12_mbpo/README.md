# Model-based Policy Optimization (MBPO)

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
