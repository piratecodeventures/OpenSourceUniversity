# Robotics Track: Capstone - Autonomous Navigation (Deep Dive)

## 📜 Story Mode: The Pathfinder

> **Mission Date**: 2043.12.05
> **Location**: Nuclear Reactor Decommission Zone
> **Officer**: Safety Drone
>
> **The Goal**: Enter Room A, find the Radioactive Barrel, pick it up, and place it in Lead Box B.
> **The Challenge**: Obstacles (Rubble). Moving People. No GPS.
>
> **The Build**: **The Nav2 Stack Integration**.

---

## 1. Project Requirements

### 1.1 Core Specs
*   **Hardware**: TurtleBot 4 (or Simulation in Gazebo).
*   **OS**: Ubuntu 22.04 + ROS 2 Humble.
*   **Sensors**: LiDAR (RPLidar), Camera (Oak-D), IMU.
*   **Task**: Mapping $\to$ Localization $\to$ Navigation $\to$ Manipulation.

### 1.2 The Tech Stack
*   **SLAM**: `slam_toolbox` (Lifelong Mapping).
*   **Navigation**: `Nav2` (Behavior Trees).
*   **Perception**: `YOLOv8` (Barrel Detection).
*   **Planning**: `A*` (Global), `DWA` (Local).

---

## 2. Architecture: ROS 2 Navigation

```mermaid
graph TD
    LiDAR --> |Scan| Costmap[Costmap 2D]
    Camera --> |Image| YOLO[Object Detector]
    
    YOLO --> |"Barrel at (X,Y)"| BehaviorTree[Behavior Tree]
    
    BehaviorTree --> |"Navigate to Pose"| GlobalPlanner[NavFn (A*)]
    GlobalPlanner --> |Path| LocalPlanner[Controller (DWA)]
    
    LocalPlanner --> |Cmd_Vel (Twist)| Wheels
    
    Costmap --> LocalPlanner
```

### 2.1 The Behavior Tree (The Brain)
XML-based logic flow:
1.  **Sequence**:
    *   Find Barrel? (Failure $\to$ Rotate Search).
    *   Compute Path.
    *   Follow Path.
    *   Recovery (If stuck $\to$ Back up).

---

## 3. The Code: Custom Behavior Node

```python
import rclpy
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

def send_goal(navigator, x, y):
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.pose.position.x = x
    goal_pose.pose.position.y = y
    
    navigator.goToPose(goal_pose)
    
    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        # Check battery / timeout
        if feedback.distance_remaining < 0.5:
             print("Approaching Target...")
             
    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print("Goal Reached!")
    else:
        print("Failed!")

def mission_logic():
    # 1. Undock
    undock()
    # 2. Go to Inspection Point
    send_goal(nav, 5.0, 2.0)
    # 3. Detect
    if detect_radiation():
        mark_hazard()
    # 4. Return
    send_goal(nav, 0.0, 0.0)
```

---

## 4. Evaluation Strategy

### 4.1 Success Metrics
1.  **Success Rate**: 10/10 runs complete without collision.
2.  **Localization Accuracy**: Final pose error < 5cm.
3.  **Real-Time Factor**: Control loop runs at > 20Hz.

### 4.2 Failure Modes (The "Edge Cases")
*   **The kidnapping**: Pick up robot and move it. Does AMCL recover?
*   **Dynamic Obstacle**: Person jumps in front. Does DWA stop in time?
*   **Glass Walls**: LiDAR sees through glass. Need Ultrasound/Depth?

---

## 5. Deployment Checklist
*   [ ] TF Tree (Transforms) correct (base_link $\to$ lidar).
*   [ ] Odometry covariance tuned (Kalman Filter).
*   [ ] Inflation Radius > Robot Radius (Don't scrape walls).
