# SAC

This exercise implements a [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (SAC).
Pseudocode: see [OpenAI's SpiningUp](https://spinningup.openai.com/en/latest/algorithms/sac.html).

- **SAC** compares with **TD3**:
1. **Entropy regularization** – encourages exploration by maximizing both reward and policy entropy.
2. **Stochastic policies** – learns a probability distribution over actions instead of deterministic ones.
3. **Automatic temperature tuning** – adjusts the trade-off between exploration and exploitation dynamically.


## Files
- `sac_exercise.py`: SAC `EXERCISE`.
- `config_pendulum.py`: Configuration file for Pendulum-v1 environment
- `config_walker.py`: Configuration file for Walker2d-v5 environment

Different games:
1. `Pendulum-v1`, a easy game is used for verifying the implementation
2. `Walker2d-v5`, a harder game, you should train it to have a good score.


## Command
Train (change `pendulum` to `walker` if needed)
```bash
# Train the model on Pendulum-v1
python practice/cli.py --config practice/exercise9_sac/config_pendulum.py
```

Play with trained model and generate video
```bash
python practice/cli.py --config practice/exercise9_sac/config_pendulum.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise9_sac/config_pendulum.py --mode push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise9_sac/config_pendulum.py --mode push_to_hub --username myuser --skip_play
```

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise9_sac/tests/ -v
```

## Comparison between SAC and TD3
| Aspect               | SAC                          | TD3                                |
|----------------------|------------------------------|------------------------------------|
| Release              | Jan 4 2018                   | Feb 26 2018                        |
| Objective            | Return + entropy             | Return only                        |
| Critic               | Double‑Q (min of 2)          | Double‑Q + smoothed target noise   |
| Policy Update        | Every step                   | Delayed (e.g. every 2 steps)       |
| Exploration          | Stochastic (auto‑tuned α)    | Deterministic + Gaussian noise     |
| Hyperparams          | Auto‑tune α                  | Manual noise & delay tuning        |
| Stability            | High (less hyperparam tuning)|	High (sensitive to noise and delay)|
| Sample Efficiency    | Strong                       | Faster early learning              |
| Final Returns        | Strong asymptotic performance| Sharp optima, risk overfitting     |
