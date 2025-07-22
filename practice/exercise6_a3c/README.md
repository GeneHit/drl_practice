# A3C

This exercise implements a [Asynchronous Advantage Actor-Critic](https://arxiv.org/abs/1602.01783) (A3C), using the previous exercise A2C+GAE.

## Files
- `a3c_cli`: the command entry file.
- `a3c_exercise.py`: A3C `EXERCISE`.
- `config_cartpole.py`: Configuration file for CartPole-v1 environment
- `config_lunar_1d.py`: Configuration file for LunarLander-v3 environment

Different games:
1. `CartPole`, a easy game is used for verifying the implementation
2. `LunarLander`, a harder game, you should train it to have a good score.

## Command
Train (change `cartpole` to `lunar_1d` if needed):
```bash
# Train the model on CartPole-v1
python practice/exercise6_a3c/a3c_cli.py --config practice/exercise6_a3c/config_cartpole.py
```

Play with trained model and generate video
```bash
python practice/cli.py --config practice/exercise6_a3c/config_cartpole.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise6_a3c/config_cartpole.py --mode push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise6_a3c/config_cartpole.py --mode push_to_hub --username myuser --skip_play
```

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise6_a3c/tests/ -v
```

## Parameter Tuning
### LunarLander
1. try: (6 worker, 6 VecEnv) -> (6 worker, 1 VecEnv) -> (6 worker, 2 VecEnv)
2. increase the global_step to 300K from 200K.


## When to Use A3C vs A2C

### 1. A3C Strengths
- **CPU‑only, multi‑core**: Hogwild-style, lock‑free updates scale across cores without a replay buffer.
- **Single‑GPU experiments**: If you prefer an asynchronous workflow or already have a multiprocess setup, each worker can `to(device)` independently.

### 2. A3C Limitations
- **Multi‑GPU / distributed**: Asynchronous updates introduce parameter staleness, gradient conflicts and communication overhead.
- **Stability & reproducibility**: Harder to debug and tune compared to synchronous algorithms.

### 3. Practical Recommendations
- **CPU clusters**: Use A3C with `model.share_memory()` + `torch.multiprocessing`.
- **Single‑GPU**:
  - _Sync option_: A2C/PPO + vectorized env (e.g. `gym.vector`, SB3) for highest utilization.
  - _Async option_: A3C per‑worker `to("cuda")` or `to("mps")`, but watch for cross‑process device copies.
- **Multi‑GPU / multi‑node**: Prefer **A2C/PPO + DDP** or frameworks like Ray RLlib / IMPALA for robust scaling.

### 4. Summary
- **A3C**: Lightweight, lock‑free, ideal for CPU or small‑scale GPU settings.
- **A2C/PPO**: Synchronous, easier to tune, best for GPU‑rich or distributed training.
