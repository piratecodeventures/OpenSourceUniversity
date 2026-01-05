# Robotics Track: Robot Learning & Sim-to-Real (Deep Dive)

## 📜 Story Mode: The Baby Robot

> **Mission Date**: 2043.11.20
> **Location**: The Holodeck
> **Officer**: Learning Specialist Vex
>
> **The Problem**: Writing `if/else` rules for walking on uneven terrain is impossible.
> There are too many edge cases.
>
> **The Solution**: **Reinforcement Learning**.
> We will put the robot in a simulation.
> We will tell it: "Move forward = +1 Point. Fall over = -10 Points."
> Let it try 1 Million times.
>
> *"Computer. Spawn Terrain Generator. Maximum speed. Train Policy."*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**: Replacing manual control engineering with Neural Networks trained by trial-and-error.
2.  **WHY**: Classical control fails in unstructured/dynamic environments (walking on rocks).
3.  **WHEN**: You have a good simulator (`Isaac Sim`/`MuJoCo`) and High-Compute.
4.  **WHERE**: `Stable-Baselines3`, `Gymnasium`, `Isaac Lab`.
5.  **WHO**: Sergey Levine (Robotic RL), Chelsea Finn (Meta-Learning).
6.  **HOW**: Simulation $\to$ Domain Randomization $\to$ Real World Deployment.

---

## 2. Mathematical Deep Dive: Policy Gradients

### 2.1 PPO (Proximal Policy Optimization)
We want to maximize Expected Reward $J(\theta) = \mathbb{E}[R]$.
Gradient Ascent: $\nabla J \approx \hat{A}_t \nabla \log \pi_\theta(a_t|s_t)$.
**The PPO Clip**:
Don't change the policy too much in one step.
$$ L = \min( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t ) $$
*   Stable, easy to tune. The "default" for robotics.

### 2.2 Domain Randomization (The Sim-to-Real Key)
We don't know the exact friction of the real floor ($\mu_{real}$).
So we train on a distribution $\mu \sim U(0.1, 0.9)$.
The Policy learns to be robust to *any* friction in that range.
To the Neural Net, the Real World is just "Another Simulation roll-out".

---

## 3. The Ship's Code (Polyglot: PPO in Isaac Sim)

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# LEVEL 2: Training a Walker
def make_env():
    # Ant-v4 (MuJoCo Quadruped)
    return gym.make("Ant-v4")

def train_robot():
    # 1. Parallel Environments (Speed up data collection)
    env = SubprocVecEnv([make_env for _ in range(8)])

    # 2. PPO Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )

    # 3. Train
    print("Training Started...")
    model.learn(total_timesteps=1_000_000)
    
    # 4. Save
    model.save("ant_walking_policy")
    print("Training Complete.")

    return model
```

---

## 4. System Architecture: The Learning Loop

```mermaid
graph TD
    Sim[Isaac Gym (GPU PhysX)] --> |State (Joints + IMU)| Policy[Neural Network]
    Policy --> |Action (Positions)| Sim
    Sim --> |Reward (Velocity - Energy)| Policy
    
    subgraph "Sim-to-Real Transfer"
        Policy -- "Weights (model.pt)" --> RealRobot[Jetson AGX]
        RealRobot --> |Inference (50Hz)| Motors
    end
```

---

## 13. Industry Interview Corner

### ❓ Real World Questions

**Q1: "Why do we use History Buffers in Robot Learning?"**
*   **Answer**: "A single frame (State) doesn't show velocity or acceleration properly (Observation is partially observable). We stack the last 10 frames ($s_{t}, s_{t-1} \dots s_{t-10}$). This allows the MLP/Transformer to infer unobserved states like 'Is my leg slipping?'"

**Q2: "What is the difference between On-Policy (PPO) and Off-Policy (SAC)?"**
*   **Answer**: "**On-Policy (PPO)** learns only from data collected by the *current* network. It throws data away. Sample Inefficient but Stable. **Off-Policy (SAC/DQN)** uses a Replay Buffer to learn from *past* data. Sample Efficient but harder to tune. For Sim (cheap data), we prefer PPO. For Real (expensive data), we prefer SAC."

---

## 14. Debug Your Thinking (Misconceptions)

> [!WARNING]
> **"End-to-End (Pixels to Torques) is standard."**
> *   **Correction**: No. Processing pixels is slow. We usually use a **Vision Module** (YOLO/SLAM) to output object coordinates, then feed Coordinates to the **Policy**. This is "Asymmetric Actor-Critic" (Train with privileged info, Act with sensors).

> [!WARNING]
> **"Sim-to-Real is solved."**
> *   **Correction**: It's the hardest problem. Soft bodies (cloth, liquids), complex contact dynamics, and lighting are still nightmares to simulate perfectly.
