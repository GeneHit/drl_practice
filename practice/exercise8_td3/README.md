# TD3

This exercise implements a [Twin Delayed DDPG](https://arxiv.org/pdf/1802.09477) (TD3).
- Pseudocode: see OpenAI's [SpiningUp](https://spinningup.openai.com/en/latest/algorithms/td3.html).
- More: [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) (DDPG)

- **TD3** improves upon **DDPG** by introducing:
  - Twin Q-networks to reduce overestimation
  - Delayed policy updates for stability
  - Target policy smoothing to reduce Q-function errors

## Files
- `td3_exercise.py`: TD3 `EXERCISE`.
- `config_pendulum.py`: Configuration file for Pendulum-v1 environment
- `config_walker.py`: Configuration file for Walker2d-v5 environment

Different games:
1. `Pendulum-v1`, a easy game is used for verifying the implementation
2. `Walker2d-v5`, a harder game, you should train it to have a good score.


## Command
Train (change `pendulum` to `walker` if needed)
```bash
# Train the model on CartPole-v1
python practice/cli.py --config practice/exercise8_td3/config_pendulum.py
```

Play with trained model and generate video
```bash
python practice/cli.py --config practice/exercise8_td3/config_pendulum.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise8_td3/config_pendulum.py --mode push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise8_td3/config_pendulum.py --mode push_to_hub --username myuser --skip_play
```

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise8_td3/tests/ -v
```

## Parameter Tuning
### CartPole


## DDPG vs TD3 Comparison

| Feature                     | DDPG                                             | TD3 (Twin Delayed DDPG)                                       |
|-----------------------------|--------------------------------------------------|----------------------------------------------------------------|
| **Number of Q Networks**    | Single Q-network                                 | ✅ Twin Q-networks (`Q1`, `Q2`), use `min(Q1, Q2)`              |
| **Actor Update Frequency**  | Every training step                              | ✅ Delayed actor updates (e.g., once every 2 critic steps)      |
| **Target Policy Smoothing** | ❌ None                                           | ✅ Adds clipped noise to target actions (target smoothing)      |
| **Q-Value Estimation Bias** | Prone to overestimation                          | ✅ Mitigates overestimation bias via twin critics               |
| **Training Stability**      | Less stable, sensitive to hyperparameters        | ✅ More stable and robust training                              |
| **Exploration**             | Action noise (e.g., OU or Gaussian)              | Same action noise + smoother targets                           |
| **Ease of Convergence**     | May struggle with convergence in complex tasks   | ✅ Better convergence and performance in continuous domains     |
| **Recommended For**         | Simple continuous control tasks                  | ✅ Most continuous control tasks with high-dimensional actions  |
| **Implementation Complexity** | Simple and lightweight                          | Slightly more complex (2 critics + delayed updates)             |

> **TD3 = DDPG + 3 Key Improvements → Better performance and robustness**
