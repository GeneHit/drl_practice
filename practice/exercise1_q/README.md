# Exercise 1: Q-Learning Implementation

Q-learning is a model-free reinforcement learning algorithm that learns the quality of actions, telling an agent what action to take under what circumstances. It does not require a model of the environment and can handle problems with stochastic transitions and rewards.


File Structure:
```
practice/exercise1_q/
├── q_table_exercise.py      # Main Q-learning `EXERCISE`
├── config_taxi.py           # Taxi environment configuration
├── config_frozen_lake.py    # FrozenLake environment configuration
├── tests/
│   └── test_q_table_train.py # Comprehensive test suite
└── README.md                # This file
```


## Command

Train the Q-learning agent using the CLI:
```bash
# Train on Taxi environment
python practice/cli.py --config practice/exercise1_q/config_taxi.py

# Train on FrozenLake environment
python practice/cli.py --config practice/exercise1_q/config_frozen_lake.py
```

Test the trained agent:

```bash
# Play with trained agent and generate video
python practice/cli.py --config practice/exercise1_q/config_taxi.py --mode play
```

Push to hub

```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise1_q/config_taxi.py --push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise1_q/config_taxi.py --push_to_hub --username myuser --skip_play
```
**Replace `myuser` with your HuggingFace account name.**

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise1_q/tests/ -v

# Run specific test
python -m pytest practice/exercise1_q/tests/test_q_table_train.py::TestQTableTraining::test_q_table_train_basic_flow -v
```


## Key Features

- **Tabular Q-learning**: Uses a Q-table to store state-action values
- **Epsilon-greedy exploration**: Balances exploration and exploitation
- **Exponential epsilon decay**: Reduces exploration over time
- **Discrete environment support**: Works with environments having discrete observation and action spaces
- **Checkpoint support**: Can resume training from saved Q-tables
- **Comprehensive evaluation**: Includes performance metrics and video recording
- **Framework integration**: Fully integrated with the practice framework

## Algorithm Components

Q-Table
- Stores Q-values for all state-action pairs
- Updated using the Bellman equation: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- Saved as pickle files for persistence

Epsilon-Greedy Policy
- **Exploration**: Random action with probability ε
- **Exploitation**: Greedy action (argmax Q(s,a)) with probability 1-ε
- **Decay**: ε decreases exponentially over episodes

Training Process
1. Initialize Q-table with zeros
2. For each episode:
   - Reset environment
   - For each step:
     - Choose action using epsilon-greedy policy
     - Execute action and observe reward and next state
     - Update Q-table using Bellman equation
     - Update state
   - Decay epsilon


## Environment Support

The implementation supports discrete environments with:
- **Discrete observation space**: States represented as integers
- **Discrete action space**: Actions represented as integers
- **Episodic tasks**: Episodes have clear start and end

### Supported Environments
- **Taxi-v3**: 500 states, 6 actions (pickup/dropoff passengers)
- **FrozenLake-v1**: 16 states (4x4 grid), 4 actions (up/down/left/right)

### Debug Tips

- Monitor epsilon decay in TensorBoard
- Check Q-table convergence by examining value changes
- Verify environment reset and step functions
- Use smaller environments for debugging (e.g., FrozenLake 4x4)

## Future Enhancements

- **Multi-step Q-learning**: N-step returns for faster learning
- **Double Q-learning**: Reduce overestimation bias
- **Prioritized experience**: Focus on important transitions
- **Function approximation**: Support for continuous/large state spaces
- **Distributed training**: Parallel environment execution
