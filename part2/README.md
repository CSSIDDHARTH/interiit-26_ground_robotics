<div align="center">

# 🦾 Part 2 — Robotic Manipulation with Reinforcement Learning

![PyBullet](https://img.shields.io/badge/Simulation-PyBullet-orange?style=for-the-badge)
![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-9cf?style=for-the-badge)
![Stable Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3%20PPO-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Kuka](https://img.shields.io/badge/Robot-Kuka%20IIwa%207DOF-lightgrey?style=for-the-badge)

**Curriculum-based Reinforcement Learning for a 7-DOF Kuka IIwa manipulator — reaching, pick-and-place, and obstacle-aware manipulation.**

*Part of the Inter IIT Tech Meet 15.0 — Ground Robotics Submission*
[← Back to root](../README.md)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Files in This Directory](#-files-in-this-directory)
- [Phase Structure](#-phase-structure)
  - [Phase 1 — Reaching](#phase-1--reaching)
  - [Phase 2 — Pick and Place](#phase-2--pick-and-place)
  - [Phase 3 — Obstacle-Aware Manipulation](#phase-3--obstacle-aware-manipulation)
  - [Phase 4 — Peg-in-Hole Insertion (Not Achieved)](#phase-4--peg-in-hole-insertion-not-achieved)
- [Technical Architecture](#-technical-architecture)
  - [Observation & Action Spaces](#observation--action-spaces)
  - [Reward Engineering](#reward-engineering)
  - [Kinematic Sanity Checks](#kinematic-sanity-checks)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Execution Instructions](#-execution-instructions)
- [Results Summary](#-results-summary)
- [Honest Assessment](#-honest-assessment)

---

## 🎯 Overview

Part 2 trains a Reinforcement Learning policy to control a **Kuka IIwa 7-DOF robotic arm** in a PyBullet physics simulation. Tasks are structured as a **curriculum of increasing difficulty** across 4 phases — with Phases 1–3 successfully converging and Phase 4 (precision peg-in-hole insertion) attempted but not solved.

---

## 📁 Files in This Directory

| File | Phase | Role |
|------|-------|------|
| `manipulator_env.py` | Phase 1 | Gymnasium environment — end-effector reaching task |
| `pick_place_env(1).py` | Phase 2 | Gymnasium environment — pick and place (version 1) |
| `pick_place_env(2).py` | Phase 2 | Gymnasium environment — pick and place (version 2, tuned) |
| `obstacle_env(1).py` | Phase 3 | Gymnasium environment — obstacle-aware manipulation (version 1) |
| `obstacle_env(2).py` | Phase 3 | Gymnasium environment — obstacle-aware manipulation (version 2, tuned) |
| `train_reach.py` | Phase 1 | PPO training script for reaching |
| `train_pick_place.py` | Phase 2 | PPO training script for pick and place |
| `train_obstacle.py` | Phase 3 | PPO training script for obstacle-aware manipulation |
| `resume_obstacle.py` | Phase 3 | Resume training from a saved checkpoint |
| `test_reach.py` | Phase 1 | Evaluate reaching policy |
| `test_pick_place.py` | Phase 2 | Evaluate pick-and-place policy |
| `test_obstacle.py` | Phase 3 | Evaluate obstacle-aware policy |

---

## 📚 Phase Structure

### Phase 1 — Reaching

**Goal:** Train the end-effector to converge to a randomly sampled 3D target position.

**Environment:** `manipulator_env.py`

**Approach:**
- Pure kinematic control — no object interaction
- Dense reward: negative L2 distance from end-effector to target
- Success bonus triggered when distance drops below a proximity threshold
- Episode terminates on success or timeout

**Key Design:**
- Distance-based reward eliminates sparse-reward exploration bottleneck
- Continuous joint velocity control (`7-DOF`) for smooth trajectories
- Randomized target positions across episodes for generalization

**Status:** ✅ Convergent policy. Reliable sub-3cm reaching across randomized targets.

---

### Phase 2 — Pick and Place

**Goal:** Extend Phase 1 with peg localization, gripper closure, and stable grasping.

**Environments:** `pick_place_env(1).py`, `pick_place_env(2).py`  
**Training:** `train_pick_place.py` | **Evaluation:** `test_pick_place.py`

**Approach:**
- Observation augmented with peg world position and contact state
- Two-stage reward: (1) approach peg centroid, (2) close gripper with contact constraint
- Gripper modeled as the 8th continuous action dimension
- Gripper closes automatically once end-effector enters a proximity threshold

**Key Design:**
- Staged reward activation: grasp reward only activates after approach is satisfied
- Peg pose queried each step via PyBullet `getBasePositionAndOrientation`
- Two environment versions: `(1)` baseline, `(2)` tuned with tighter proximity thresholds and adjusted reward weights

**Status:** ✅ Stable grasp behavior. Agent reliably picks peg from table surface.

---

### Phase 3 — Obstacle-Aware Manipulation

**Goal:** Execute pick-and-place in the presence of static environmental obstacles (vertical barriers).

**Environments:** `obstacle_env(1).py`, `obstacle_env(2).py`  
**Training:** `train_obstacle.py` | **Resume:** `resume_obstacle.py` | **Evaluation:** `test_obstacle.py`

**Approach:**
- Static URDF obstacles added to PyBullet scene (vertical barriers, walls)
- Hard collision penalty: large negative reward on any robot-obstacle contact
- Soft proximity penalty: additional negative shaping as end-effector approaches obstacle bounds
- Curriculum initialization: obstacles initially placed away from the trajectory, progressively moved closer

**Key Design:**
- Two environment versions: `(1)` basic obstacles, `(2)` tighter clearances and stronger collision shaping
- `resume_obstacle.py` allows training to continue from a saved checkpoint — critical for long Phase 3 runs
- PPO's clipped objective prevents catastrophic forgetting of Phase 2 pick behavior during Phase 3 fine-tuning

**Status:** ✅ Agent navigates around static obstacles to complete pick-and-place. Collision rate drops significantly over training.

---

### Phase 4 — Peg-in-Hole Insertion (Not Achieved)

**Goal:** Insert the grasped peg into a target hole with sub-millimeter precision alignment.

**What Was Attempted:**
- Extended Phase 3 environment with a hole fixture model in PyBullet
- Dense reward: peg-tip to hole-mouth distance + angular misalignment penalty + insertion depth reward
- Contact detection to measure peg entry into hole
- Force-compliance reward to encourage smooth insertion
- Multiple reward formulations and hyperparameter sweeps

**Why It Failed:**

> We tried our best on Phase 4. Despite extensive effort across multiple reward designs, curriculum strategies, and PPO configurations, **the success rate could not exceed ~10% (0.1)**. We made the decision to stop at this point and submit our completed work rather than continuing without convergence.

| Root Cause | Detail |
|------------|--------|
| Precision requirements | Sub-mm alignment makes the success region an infinitesimal fraction of state space — dense shaping alone is insufficient |
| Contact dynamics instability | PyBullet near-contact forces are discontinuous and noisy, destabilizing PPO gradient updates in the insertion zone |
| Compounding upstream error | Any grasp pose imprecision from Phase 2/3 directly amplifies at Phase 4 insertion — end-to-end error accumulates |
| Exploration gap | RL random exploration cannot efficiently discover the narrow insertion corridor without demonstrations or guided sampling |

**Status:** ⚠️ Attempted — best observed success rate: **~10% (0.1)**. Phase 4 not included in final submission.

> Phase 4 likely requires techniques beyond vanilla PPO — such as **imitation learning from demonstrations**, **force/torque feedback** in the observation space, or a **task-space compliance controller** with tightly constrained insertion-phase action bounds.

---

## 🏗️ Technical Architecture

### Observation & Action Spaces

```
┌─────────────────────────────────────────────────────────────┐
│                   OBSERVATION SPACE (18D)                   │
├──────────────────────────────┬──────┬───────────────────────┤
│  Joint Positions             │  7D  │  θ₁ … θ₇             │
│  End-Effector Position       │  3D  │  (x, y, z)            │
│  End-Effector Orientation    │  3D  │  (roll, pitch, yaw)   │
│  Peg Position                │  3D  │  (x_p, y_p, z_p)      │
│  Relative Vector (EE → Peg)  │  2D  │  (Δx, Δy)             │
└──────────────────────────────┴──────┴───────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               ACTION SPACE (8D, Continuous [-1, 1])         │
├──────────────────────────────┬──────┬───────────────────────┤
│  Joint Velocity Commands     │  7D  │  δθ₁ … δθ₇            │
│  Gripper Command             │  1D  │  0.0 = open, 1.0 = close│
└──────────────────────────────┴──────┴───────────────────────┘
```

Actions are bounded in `[-1, 1]` and scaled internally to joint velocity limits.

---

### Reward Engineering

Dense reward shaping is the core design principle across all phases to avoid sparse-reward training failure:

```python
# ── Phase 1: Reaching ────────────────────────────────────────
r = - α * dist(end_effector, target)
    + β * success_bonus              # fires when dist < threshold
    - γ * timestep_penalty           # encourages faster convergence

# ── Phase 2: Pick & Place ────────────────────────────────────
r = - α * dist(end_effector, peg)
    + β * grasp_contact_reward       # activates on gripper contact
    - γ * timestep_penalty

# ── Phase 3: Obstacle-Aware ──────────────────────────────────
r = (Phase 2 reward)
    - δ * is_in_collision()          # hard penalty on contact
    - ε * obstacle_proximity()       # soft shaping near obstacles

# ── Phase 4: Insertion (attempted) ───────────────────────────
r = - α * dist(peg_tip, hole_mouth)
    - β * angular_misalignment(peg, hole_axis)
    + γ * insertion_depth            # depth reward via contact sensing
    - δ * is_in_collision()
```

**Principles:**
- No sparse terminal rewards — every state transition provides a gradient signal
- Staged activation: later reward terms only activate when upstream conditions are met
- Collision penalties are order-of-magnitude larger than progress rewards to enforce safety

---

### Kinematic Sanity Checks

Before committing to RL training, IK solvability was validated across the full range of task configurations to confirm physical feasibility:

```python
# Validates that the target space is kinematically reachable before training
def run_kinematic_sanity_check(n_trials=1000, clearance_range=(0.01, 0.05)):
    success = 0
    for _ in range(n_trials):
        target = sample_target_position(clearance_range)
        ik_sol = pybullet.calculateInverseKinematics(robot_id, EE_LINK, target)
        if validate_ik_solution(ik_sol, target, tol=1e-3):
            success += 1
    print(f"IK Solvability Rate: {success / n_trials * 100:.1f}%")
    # Result: >90% solvability confirmed across all phases
```

High solvability rates (>90%) were confirmed before each phase's training began — ensuring that training failures are attributable to policy learning, not environment infeasibility.

---

## 🛠️ Tech Stack

| Domain | Tool | Version |
|--------|------|---------|
| Physics Simulation | PyBullet | 3.x |
| RL Environment | Gymnasium | 0.29.x |
| RL Algorithm | Stable Baselines3 — PPO | 2.x |
| Robot Model | Kuka IIwa 14 | URDF (built-in PyBullet) |
| Language | Python | 3.10+ |
| Training Monitor | TensorBoard | 2.x |
| Compute | NVIDIA GPU (CUDA) | — |

---

## ⚙️ Installation & Setup

```bash
cd ~/ros2_ws/PART_2_MANIPULATOR_RL/
# (or wherever part2/ lives)

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pybullet gymnasium stable-baselines3 torch tensorboard numpy matplotlib
```

---

## ▶️ Execution Instructions

### Training

```bash
source venv/bin/activate

# Phase 1 — Reaching
python train_reach.py

# Phase 2 — Pick & Place
python train_pick_place.py

# Phase 3 — Obstacle-Aware
python train_obstacle.py

# Phase 3 — Resume from checkpoint
python resume_obstacle.py
```

### Evaluation

```bash
# Evaluate Phase 1 reaching policy
python test_reach.py

# Evaluate Phase 2 pick-and-place policy
python test_pick_place.py

# Evaluate Phase 3 obstacle-aware policy
python test_obstacle.py
```

### Monitor Training

```bash
tensorboard --logdir logs/
# Open http://localhost:6006 in browser
```

---

## 📊 Results Summary

| Phase | Task | Environment File | Success Rate | Status |
|-------|------|-----------------|-------------|--------|
| Phase 1 | End-Effector Reaching | `manipulator_env.py` | ~95%+ | ✅ Converged |
| Phase 2 | Pick & Place | `pick_place_env(2).py` | ~85%+ | ✅ Converged |
| Phase 3 | Obstacle-Aware Pick & Place | `obstacle_env(2).py` | ~75%+ | ✅ Converged |
| Phase 4 | Precision Peg-in-Hole Insertion | *(not included)* | **~10% (max)** | ⚠️ Not Achieved |

---

## ⚠️ Honest Assessment

### ✅ What We Achieved

- A complete **3-phase RL curriculum** using custom Gymnasium environments and PPO
- Clean **18D observation space** and **8D continuous action space** with kinematic pre-validation
- **Dense reward engineering** that successfully eliminated sparse-reward failures across Phases 1–3
- Two tuned environment versions per phase, with `resume_obstacle.py` enabling long training runs
- Reliable, convergent policies for reaching, grasping, and obstacle-aware manipulation

### ❌ What We Could Not Achieve

Phase 4 (precision peg-in-hole insertion) was not solved. Despite our best efforts — multiple reward designs, curriculum modifications, and hyperparameter sweeps — the success rate **never exceeded 10%**. We tried our best and made the decision to stop here rather than continuing without measurable progress.

---

<div align="center">

[← Back to root README](../README.md) | [Part 1 — Navigation & SLAM](../part%201/README.md)

*Inter IIT Tech Meet 15.0 — Ground Robotics*

</div>
