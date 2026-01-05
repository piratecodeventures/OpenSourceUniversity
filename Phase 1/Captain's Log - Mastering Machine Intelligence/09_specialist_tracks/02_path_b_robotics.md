# Path B: Robotics & Embodied AI (The Pilot)

## 📜 Career Profile: The Pilot
*   **Role**: Robotics Engineer, Computer Vision Scientist, SLAM Engineer.
*   **Mission**: Bridge the "Sim-to-Real Gap". Make software move hardware without breaking it.
*   **Income Potential**: $180k - $400k.
*   **Core Stack**: ROS 2, C++, Python, Isaac Sim, PCL (Point Cloud Library).

---

## 📅 Sprint 11: Computer Vision Mastery (Weeks 41-44)

> **Theme**: "The Robot Eye"

### 11.1 3D Geometry & Math
*   **11.1.1 Projective Geometry**:
    *   **Pinhole Model**: $x = K [R|t] X$. Intrinsics ($K$) vs Extrinsics ($R, t$).
    *   **Homogeneous Coordinates**: Adding a dimension (w=1) to handle translation as matrix multiplication.
*   **11.1.2 Epipolar Geometry**:
    *   Stereo Vision. The "Essential Matrix" $E$ and "Fundamental Matrix" $F$.
    *   Triangulation: Finding depth $Z$ from disparity $d$.

### 11.2 Visual SLAM (Simultaneous Localization & Mapping)
*   **11.2.1 Frontend (Visual Odometry)**:
    *   Feature Extraction (ORB, SIFT). 
    *   Feature Matching (FLANN, Brute Force).
    *   PnP (Perspective-n-Point) problem.
*   **11.2.2 Backend (Optimization)**:
    *   **Bundle Adjustment**: Minimizing reprojection error globally ($argmin \sum ||x - P X||^2$).
    *   **Loop Closure**: "Have I been here before?" (Bag of Words).
    *   **Graph SLAM**: Pose Graph Optimization (g2o).

### 11.3 Deep Perception
*   **11.3.1 Object Detection**: YOLOv8 (Real-time).
*   **11.3.2 Instance Segmentation**: Mask2Former (Pixel-level understanding).
*   **11.3.3 NeRF (Neural Radiance Fields)**: Representing scenes as Neural Networks ($XYZ \to RGB\sigma$).

---

## 📅 Sprint 12: Robotics Fundamentals (Weeks 45-48)

> **Theme**: "The Mathematical Body"

### 12.1 Kinematics (Computation of Motion)
*   **12.1.1 Forward Kinematics (FK)**:
    *   DH Parameters (Denavit-Hartenberg).
    *   Chain multiplication from Base to End-Effector.
*   **12.1.2 Inverse Kinematics (IK)**:
    *   **Analytical**: Geometric solution (Fast, specific to robot).
    *   **Numerical**: Jacobian Inverse ($J^\dagger$). Iterative Newton-Raphson.
    *   *Singularities*: Where Determinant($J$) = 0. Use Damped Least Squares.

### 12.2 Dynamics (Forces & Torques)
*   **12.2.1 Lagrangian Formulation**:
    *   $L = T - V$ (Kinetic - Potential Energy).
    *   Equation of Motion: $M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = \tau$.
*   **12.2.2 Control Theory**:
    *   **PID**: Proportional, Integral, Derivative.
    *   **MPC (Model Predictive Control)**: Optimizing a trajectory over a finite horizon.

### 12.3 Motion Planning
*   **12.3.1 Graph Search**: A*, Dijkstra (Grid based).
*   **12.3.2 Sampling Based**:
    *   **RRT (Rapidly-exploring Random Tree)**: Exploring high-dimensional configuration space (C-Space).
    *   **RRT***: Converges to optimal path.

---

## 📅 Sprint 13: Robot Learning (Weeks 49-52)

> **Theme**: "The Neural Brain"

### 13.1 Imitation Learning (Behavioral Cloning)
*   **13.1.1 Teleoperation**: Recording Expert demonstrations $(s, a)$.
*   **13.1.2 The Distribution Shift**:
    *   DAgger (Dataset Aggregation): Expert corrects the Robot's mistakes interactively.

### 13.2 Reinforcement Learning (RL)
*   **13.2.1 Policy Gradients**:
    *   **PPO (Proximal Policy Optimization)**: Clipping updates for stability. Standard for locomotion (walking).
*   **13.2.2 Soft Actor-Critic (SAC)**:
    *   Entropy Regularization exploration. Standard for manipulation (arms).
*   **13.2.3 Reward Shaping**:
    *   Defining the function: $R = w_1 \cdot dist + w_2 \cdot energy + w_3 \cdot smooth$.

### 13.3 Sim-to-Real Transfer
*   **13.3.1 Domain Randomization**:
    *   Randomizing Friction, Mass, Lighting, Textures in Sim.
    *   Real world becomes "center of distribution".
*   **13.3.2 System Identification**: Using ML to predict physical parameters ($mass$) online.

---

## 📅 Sprint 14: Capstone - The Autonomous System (Weeks 53-56)

### 14.1 The Stack: ROS 2 Architecture
*   **14.1.1 Behavior Trees**: High-level logic (If Battery < 10% $\to$ Go Home).
*   **14.1.2 Nav2 Stack**:
    *   Global Planner (Map path).
    *   Local Planner (Obstacle avoidance - DWA/TEB).
    *   Costmaps (Static vs Inflation).

### 14.2 Hardware Integration
*   **14.2.1 Sensors**:
    *   LiDAR (2D/3D Point Clouds).
    *   IMU (Accelerotmeter/Gyro) for Odom.
    *   Cameras (RGB-D).
*   **14.2.2 Compute**: Jetson Orin / Raspberry Pi 5.
    *   Using TensorRT for edge inference.

---

## 💻 The Ship's Code: Inverse Kinematics (Jacobian)

```python
import numpy as np

def inverse_kinematics(target_pos, current_joints, link_lengths):
    """
    Numerical IK using Jacobian Pseudo-Inverse (Damped Least Squares)
    """
    lambda_val = 0.01 # Damping factor
    max_iters = 100
    tolerance = 1e-3
    
    q = np.array(current_joints)
    
    for i in range(max_iters):
        # 1. Forward Kinematics (Where are we now?)
        current_pos = forward_kinematics(q, link_lengths)
        
        # 2. Error
        error = target_pos - current_pos
        if np.linalg.norm(error) < tolerance:
            return q
            
        # 3. Compute Jacobian (Numerical or Analytical)
        J = compute_jacobian(q, link_lengths)
        
        # 4. Damped Least Squares: J.T * (J * J.T + lambda^2 * I)^-1
        # This handles singularities where J is non-invertible
        J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_val**2 * np.eye(3))
        
        # 5. Update Joints
        dq = J_dls @ error
        q += dq
        
    return q # Convergence failed or reached
```

---

## ❓ Industry Interview Corner

**Q1: "How do you handle the Kidnapped Robot Problem?"**
*   **Answer**: "The robot wakes up and doesn't know where it is. Global Localization. **Monte Carlo Localization (MCL)** (Particle Filter) spreads particles everywhere. As the robot moves and sensors see unique landmarks, particles in wrong locations die out, and particles in the right location cluster. Eventually, the cloud converges."

**Q2: "Explain the difference between Holonomic and Non-Holonomic constraints."**
*   **Answer**: "Holonomic: Can move in any direction (Omni-wheel robot, flying drone). Non-Holonomic: Constrained movement (Car). A car cannot move sideways directly; it must park parallelly. This complicates path planning (Reeds-Shepp curves)."

**Q3: "Why is Depth Sensing hard with Stereo Cameras on white walls?"**
*   **Answer**: "Stereo relies on **Feature Matching** between left/right images. A white wall has no texture features (textureless). Correspondence fails. Active Depth (LiDAR/Structured Light) is needed for textureless surfaces."
