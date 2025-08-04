# PPO + Curiosity + PyTorch DDP Distributed Reinforcement Learning Architecture


## Command
Verify the implementation without DDP.
```
python practice/cli.py --config practice/exercise10_ddp_ppo/config_reacher.py
```
Train (change `pusher` to `standup` if needed)
```bash
torchrun --nproc_per_node=3  --master_addr="127.0.0.1" --master_port=12345 practice/cli.py --config practice/exercise10_ddp_ppo/config_ddp_reacher.py
```

Play with trained model and generate video
```bash
python practice/cli.py --config practice/exercise10_ddp_ppo/config_ddp_reacher.py --mode play
```

Push to hub
```bash
# generate video and push to hub
python practice/cli.py --config practice/exercise10_ddp_ppo/config_ddp_reacher.py --mode push_to_hub --username myuser

# only push to hub
python practice/cli.py --config practice/exercise10_ddp_ppo/config_ddp_reacher.py --mode push_to_hub --username myuser --skip_play
```

Run the comprehensive test suite:
```bash
# Run all tests
python -m pytest practice/exercise10_ddp_ppo/tests/ -v
```


## 🧠 Project Goal

Design a scalable on‑policy RL system that integrates:

- ✅ **PPO (Proximal Policy Optimization)** — stable, clipped surrogate objective
- ✅ **Curiosity Module** — intrinsic reward to drive exploration (ICM / RND / NGU)
- ✅ **PyTorch DDP (DistributedDataParallel)** — multi‑GPU synchronous training
- ✅ **Environment Parallelism** — vectorized or multi‑process sampling (e.g., Pusher-v5, HumanoidStandup-v5)

### 📐 Architecture Overview

This design decouples sampling from training while leveraging both environment parallelism and multi‑GPU DDP:

1. **Parallel Environments**
   - CPU‑based vectorized or subprocess environments collect trajectories.
2. **Data Sharding**
   - Collected on‑policy trajectories are concatenated into one big batch and split across GPUs.
3. **Curiosity Module**
   - Computes intrinsic reward for each transition (e.g. ICM prediction error or RND error).
   - Intrinsic reward is combined with extrinsic reward before advantage estimation.
4. **Multi‑GPU PPO Update**
   - Wrap policy & value networks with `torch.nn.parallel.DistributedDataParallel`.
   - Each GPU takes its data shard, computes losses, calls `backward()`.
   - DDP automatically all‑reduces gradients and synchronizes parameters.
   - `optimizer.step()` is called on each GPU to update the shared model.

### 🔁 System Flow Diagram

```text
[DP 0]: Vectorized Envs (CPU) ──collect──▶ Rollout ──▶ forward/loss/backward ──all‑reduce──▶ optimizer.step()
                                                                                   |
                                                                                   |
[DP 1]: Vectorized Envs (CPU) ──collect──▶ Rollout ──▶ forward/loss/backward ──all‑reduce──▶ optimizer.step()
                                                                                   |
                                                                                   |
[DP 2]: Vectorized Envs (CPU) ──collect──▶ Rollout ──▶ forward/loss/backward ──all‑reduce──▶ optimizer.step()
...
[DP n]
```
