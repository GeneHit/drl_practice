# Exercise 1: Q-Learning

Q-learning is a model-free reinforcement learning algorithm that learns the quality of actions, telling an agent what action to take under what circumstances. It does not require a model of the environment and can handle problems with stochastic transitions and rewards.


## File Structure:
- `q_table_exercise.py`: Main `EXERCISE`.
- `config_frozen_lake.py`: FrozenLake environment configuration.
- `config_taxi.py`: Taxi environment configuration.
- `tests/test_q_table_train.py`: Comprehensive test suite


## Command

Train the Q-table agent using the CLI:
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
python practice/cli.py --config practice/exercise1_q/config_taxi.py --mode push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise1_q/config_taxi.py --mode push_to_hub --username myuser --skip_play
```
**Replace `myuser` with your HuggingFace account name.**

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise1_q/tests/ -v

# Run specific test
python -m pytest practice/exercise1_q/tests/test_q_table_train.py::TestQTableTraining::test_q_table_train_basic_flow -v
```


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

## Future Enhancements

- **Multi-step Q-learning**: N-step returns for faster learning
- **Double Q-learning**: Reduce overestimation bias
- **Prioritized experience**: Focus on important transitions
- **Function approximation**: Support for continuous/large state spaces
- **Distributed training**: Parallel environment execution
