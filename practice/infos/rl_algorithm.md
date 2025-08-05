# Reinforcement Learning Algorithms

## Comparison Table

- C: continuous; D: discrete

| Algorithm Name       | Type              | Policy Type        | Supported Action Space       | Notes & Characteristics                                                  |
|----------------------|-------------------|---------------------|-------------------------------|---------------------------------------------------------------------------|
| **DQN**              | Value-based       | Discrete (ε-greedy) | ✅ D ❌ C      | Classic deep Q-learning, best for small discrete action spaces           |
| **Double DQN**       | Value-based       | Discrete            | ✅ D ❌ C      | Reduces overestimation bias in Q-values                                  |
| **Dueling DQN**      | Value-based       | Discrete            | ✅ D ❌ C      | Separates state-value and advantage for better generalization            |
| **REINFORCE**        | Policy Gradient   | Stochastic          | ✅ D ✅ C      | Basic Monte Carlo policy gradient, high variance                         |
| **A2C**              | Actor-Critic      | Stochastic          | ✅ D ✅ C      | Uses synchronous environments and advantage estimates                    |
| **A3C**              | Actor-Critic      | Stochastic          | ✅ D ✅ C      | Asynchronous version of A2C, faster training                             |
| **PPO**              | Actor-Critic      | Stochastic (clipped)| ✅ D ✅ C      | Popular, stable, and efficient with clipping mechanism                   |
| **TRPO**             | Actor-Critic      | Stochastic          | ✅ D ✅ C      | Uses trust region constraints (KL divergence), more complex              |
| **DDPG**             | Actor-Critic      | Deterministic       | ❌ D ✅ C      | Deterministic policy, suited for continuous control                      |
| **TD3**              | Actor-Critic      | Deterministic       | ❌ D ✅ C      | Improved DDPG with twin critics and delayed updates                      |
| **SAC**              | Actor-Critic      | Stochastic (entropy)| ✅ D ✅ C      | Maximum entropy RL, stable and exploratory                               |
| **Discrete SAC**     | Actor-Critic      | Stochastic          | ✅ D ❌ C      | Adapted version of SAC for discrete action spaces                        |
| **Hybrid SAC / DDPG**| Actor-Critic      | Mixed (stoch/det)   | ✅ D ✅ C ✅ Mixed | Handles mixed action spaces (e.g., discrete + continuous parameters)   |

### Notes
- **Policy Type:**
  - *Deterministic*: directly outputs the action
  - *Stochastic*: outputs a distribution (e.g., Gaussian or Categorical)

- **Action Space:**
  - **Discrete**: finite actions (e.g., left, right)
  - **Continuous**: real-valued (e.g., steering angle, torque)
  - **Mixed**: both discrete and continuous components (e.g., "select tool + set angle")

### Algorithm Recommendations by Environment

| Environment Type                 | Recommended Algorithms      |
|----------------------------------|-----------------------------|
| Classic discrete (e.g., CartPole)| DQN / PPO / A2C             |
| Continuous control (e.g., MuJoCo)| PPO / SAC / TD3             |
| High-dimensional + exploration   | SAC / TD3                   |
| Discrete control with feedback   | PPO / A2C                   |
| Mixed action environments        | Hybrid SAC / Multi-policy   |


## When to Use A3C vs A2C (Simplified)

A3C – Strengths & Use Cases
- Great for **CPU-only, multi-core systems** (lock-free updates, no replay buffer needed).
- Works well with **single-GPU** setups using asynchronous multiprocess workflows.

A3C – Limitations
- Not ideal for **multi-GPU or distributed training** due to stale gradients and sync issues.
- **Harder to debug and tune**, less stable than synchronous methods.

Recommendations
- **CPU clusters**: Use A3C with `model.share_memory()` and `torch.multiprocessing`.
- **Single-GPU**:
  - Prefer **A2C/PPO + vectorized envs** for synchronous efficiency.
  - A3C is possible (per-process `.to("cuda")`), but avoid cross-process GPU ops.
- **Multi-GPU or multi-node**: Use **A2C/PPO + DDP**, or frameworks like **Ray RLlib** or **IMPALA**.

### Summary
- **A3C**: Lightweight and parallel-friendly for CPU or basic GPU setups.
- **A2C/PPO**: More stable, better suited for GPU-heavy or scalable systems.


## Skill for High-variant environment

common
- Normalize input features to reduce variance impact.
- Use reward scaling or clipping to stabilize training.
- Encourage exploration.
- Use frame stacking or recurrent networks for temporal consistency.
- Apply clip_grad_norm_

on-policy
- Implement reward normalization for stable policy updates.
- Increase rollout batch size to average over more transitions.
- Use value loss clip
- Normalize the returns

off-policy
- Apply prioritized experience replay to focus on informative transitions.
- Increase target network update frequency for smoother learning.
- Use huber loss or TD Error Clippi
- Better soft target updates for stability (e.g., Polyak averaging).
- Regularize Q-functions to prevent overfitting to noisy targets.
