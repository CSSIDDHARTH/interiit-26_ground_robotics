<div align="center">

# 🗺️ Part 1 — Autonomous Navigation & SLAM

![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=for-the-badge&logo=ros&logoColor=white)
![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-ff6600?style=for-the-badge)
![Nav2](https://img.shields.io/badge/Navigation-Nav2-0078d7?style=for-the-badge)
![SLAM](https://img.shields.io/badge/Mapping-SLAM%20Toolbox-brightgreen?style=for-the-badge)

**Autonomous mobile base navigation, real-time SLAM, and holonomic drive control for the competition arena.**

*Part of the Inter IIT Tech Meet 15.0 — Ground Robotics Submission*
[← Back to root](../README.md)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Files in This Directory](#-files-in-this-directory)
- [Holonomic Drive Controller](#-holonomic-drive-controller)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Execution Instructions](#-execution-instructions)
- [Capabilities](#-capabilities)

---

## 🎯 Overview

Part 1 implements the full autonomous navigation pipeline for a **holonomic (omnidirectional) ground mobile robot** operating in an unknown competition arena. The system integrates:

- A custom **holonomic drive controller** translating velocity commands into individual wheel velocities
- **SLAM Toolbox** for real-time occupancy grid map generation
- **Nav2** for global path planning and local obstacle avoidance
- **Gazebo Harmonic** as the simulation environment

---

## 📁 Files in This Directory

| File | Description |
|------|-------------|
| `task1.py` | ROS 2 node implementing the `HolonomicDriveController` — converts `/cmd_vel` Twist messages into 4-wheel velocity commands |
| `README.md` | This file |

---

## 🚗 Holonomic Drive Controller

### `task1.py` — `HolonomicDriveController` Node

This is the core low-level controller that enables **omnidirectional motion** for the competition robot. It bridges the ROS 2 navigation stack (which outputs standard `Twist` velocity commands) to the robot's 4-wheel holonomic drive system.

**Node Summary:**

| Property | Value |
|----------|-------|
| Node Name | `holonomic_drive_controller` |
| Subscribes To | `/cmd_vel` (`geometry_msgs/Twist`) |
| Publishes To | `/wheel_velocities` (`std_msgs/Float64MultiArray`) |
| Wheel Layout | 4-wheel (FL, FR, RL, RR) |

### Kinematics

The controller implements the standard **mecanum/holonomic inverse kinematics** equations to decompose a body-frame velocity command `(vx, vy, ωz)` into individual wheel velocities:

```
v_FL =  vx - vy - ωz * (lx + ly)
v_FR =  vx + vy + ωz * (lx + ly)
v_RL =  vx + vy - ωz * (lx + ly)
v_RR =  vx - vy + ωz * (lx + ly)
```

Where:
- `vx`, `vy` — linear velocity in the robot's x and y axes (m/s)
- `ωz` — angular velocity about the z-axis (rad/s)
- `lx`, `ly` — half the robot's wheelbase in x and y directions (both set to `0.2 m`)

**Velocity Clamping:**

All inputs are clamped before the kinematics computation to prevent actuator saturation:

```python
max_linear_vel  = 1.0   # m/s
max_angular_vel = 1.5   # rad/s
```

### Node Code

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

class HolonomicDriveController(Node):
    def __init__(self):
        super().__init__('holonomic_drive_controller')
        self.subscription = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.publisher = self.create_publisher(
            Float64MultiArray, '/wheel_velocities', 10)
        self.lx = 0.2   # half-wheelbase x
        self.ly = 0.2   # half-wheelbase y
        self.max_linear_vel  = 1.0
        self.max_angular_vel = 1.5

    def cmd_vel_callback(self, msg):
        vx = max(min(msg.linear.x,  self.max_linear_vel),  -self.max_linear_vel)
        vy = max(min(msg.linear.y,  self.max_linear_vel),  -self.max_linear_vel)
        wz = max(min(msg.angular.z, self.max_angular_vel), -self.max_angular_vel)

        v_fl = vx - vy - wz * (self.lx + self.ly)
        v_fr = vx + vy + wz * (self.lx + self.ly)
        v_rl = vx + vy - wz * (self.lx + self.ly)
        v_rr = vx - vy + wz * (self.lx + self.ly)

        wheel_msg = Float64MultiArray()
        wheel_msg.data = [v_fl, v_fr, v_rl, v_rr]
        self.publisher.publish(wheel_msg)
```

---

## 🏗️ System Architecture

```
                    ┌─────────────────────────────────────┐
                    │         ROS 2 Humble Ecosystem       │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
   ┌──────▼──────┐         ┌───────▼───────┐       ┌───────▼───────┐
   │  SLAM       │         │   Nav2 Stack  │       │ Gazebo        │
   │  Toolbox    │         │  (Planner +   │       │ Harmonic      │
   │  (Mapping)  │◄───────►│   Controller) │◄─────►│ (Simulation)  │
   └──────┬──────┘         └───────┬───────┘       └───────────────┘
          │                        │
          │                 ┌──────▼──────────────┐
          │                 │  /cmd_vel (Twist)   │
          │                 └──────┬──────────────┘
          │                        │
          │                 ┌──────▼──────────────────────┐
          │                 │  HolonomicDriveController   │
          │                 │        (task1.py)           │
          │                 └──────┬──────────────────────┘
          │                        │
          └────────────────►┌──────▼──────────────┐
                            │  /wheel_velocities  │
                            │  [FL, FR, RL, RR]   │
                            └─────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Tool | Notes |
|-----------|------|-------|
| Middleware | ROS 2 Humble | Ubuntu 22.04 |
| Simulation | Gazebo Harmonic | Full physics |
| Navigation Stack | Nav2 | Global + local planner |
| Mapping | SLAM Toolbox | Online async SLAM |
| Drive Controller | `task1.py` (custom) | Holonomic IK node |
| Language | Python 3.10+ | ROS 2 Python client |

---

## ⚙️ Installation & Setup

### Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Humble (full desktop install)
- Gazebo Harmonic

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Install Nav2
sudo apt install ros-humble-nav2-bringup

# Install SLAM Toolbox
sudo apt install ros-humble-slam-toolbox

# Install Gazebo ROS packages
sudo apt install ros-humble-gazebo-ros-pkgs

# Build your workspace
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

## ▶️ Execution Instructions

### Step 1 — Launch the Holonomic Drive Controller

```bash
cd ~/ros2_ws/
source install/setup.bash

# Run the controller node directly
python3 "part 1/task1.py"

# OR as a ROS 2 run (if packaged)
ros2 run <your_package> holonomic_drive_controller
```

### Step 2 — Launch Gazebo Simulation

```bash
ros2 launch gazebo_ros gazebo.launch.py world:=arena.world
```

### Step 3 — Launch SLAM Toolbox

```bash
ros2 launch slam_toolbox online_async_launch.py \
  use_sim_time:=true
```

### Step 4 — Launch Nav2

```bash
ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=true \
  params_file:=config/nav2_params.yaml
```

### Step 5 — Send a Navigation Goal

```bash
# Via CLI
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "pose: {header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 1.5, z: 0.0}}}"

# Or use RViz2 → "2D Nav Goal" tool for interactive goal setting
rviz2
```

### Verify the Controller is Running

```bash
# Check the node is alive
ros2 node list | grep holonomic

# Echo wheel velocities to verify output
ros2 topic echo /wheel_velocities

# Send a test velocity command manually
ros2 topic pub /cmd_vel geometry_msgs/Twist \
  "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {z: 0.3}}" --once
```

---

## ✅ Capabilities

| Capability | Description | Status |
|------------|-------------|--------|
| Holonomic Drive | Omnidirectional motion via mecanum kinematics | ✅ |
| Real-time SLAM | Online 2D occupancy grid mapping | ✅ |
| Autonomous Navigation | Global + local path planning via Nav2 | ✅ |
| Waypoint Tracking | Sequential goal navigation | ✅ |
| Obstacle Avoidance | Costmap-based reactive avoidance | ✅ |
| Velocity Clamping | Safety-capped actuator commands | ✅ |

---

<div align="center">

[← Back to root README](../README.md) | [Part 2 — Robotic Manipulation →](../part2/README.md)

*Inter IIT Tech Meet 15.0 — Ground Robotics*

</div>
