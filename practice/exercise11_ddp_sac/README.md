# SAC + PER + PyTorch DDP Distributed Reinforcement Learning Architecture

## 🧠 Project Goal

Design a scalable distributed reinforcement learning system that integrates:

- ✅ **SAC (Soft Actor-Critic)** — stable, off-policy continuous control
- ✅ **PER (Prioritized Experience Replay)** — prioritizes valuable experiences by TD-error
- ✅ **PyTorch DDP (DistributedDataParallel)** — multi-GPU parallel training
- ✅ Multi-environment sampling (e.g., Pusher-v5, Humanoid-v5)

### 📐 Architecture Overview

This is an Actor-Learner decoupled architecture:

- Each **DDP process** acts as an "Actor + Learner" unit
- All processes share a **centralized PER buffer** or use a **distributed PER server**.
- The model is wrapped with PyTorch DDP for **synchronized parameter updates**
- Each process samples and trains independently using batches from the PER buffer

### 🔁 System Flow Diagram

```text
[DP 0]: Vectorized Envs (CPU) ──collect──▶ Buffer ──▶ forward/loss/backward ──all‑reduce──▶ optimizer.step()
                                                                                   |
                                                                                   |
[DP 1]: Vectorized Envs (CPU) ──collect──▶ Buffer ──▶ forward/loss/backward ──all‑reduce──▶ optimizer.step()
                                                                                   |
                                                                                   |
[DP 2]: Vectorized Envs (CPU) ──collect──▶ Buffer ──▶ forward/loss/backward ──all‑reduce──▶ optimizer.step()
...
[DP n]
```
