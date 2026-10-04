import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np


class ManipulatorReachEnv(gym.Env):
    """KUKA iiwa reaching task.

    Key design choices (vs. the first version):
      * normalized action space [-1, 1], scaled inside the env
      * targets are generated from forward kinematics of a random joint
        configuration, so every target is guaranteed reachable
      * we integrate a *commanded* joint target (no gravity-sag drift from
        re-reading the measured joint position every step)
      * richer observation: q, qd, ee, target, relative vector, distance
      * dense + potential reward, and a success bonus
      * difficulty knob for curriculum learning
    """
    metadata = {"render_modes": ["human"]}

    HOME = np.array([0.0, 0.4, 0.0, -1.2, 0.0, 0.8, 0.0])
    SPREAD = np.array([1.2, 0.7, 1.2, 0.7, 1.2, 0.7, 1.2])

    def __init__(self, render_mode=None, max_steps=150, action_scale=0.05,
                 sim_substeps=10, success_radius=0.05, difficulty=1.0):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.sim_substeps = sim_substeps
        self.success_radius = success_radius
        self.difficulty = float(difficulty)

        self.cid = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0],
                                useFixedBase=True, physicsClientId=self.cid)
        self.ee_index = 6
        self.joint_indices = [
            i for i in range(p.getNumJoints(self.robot, physicsClientId=self.cid))
            if p.getJointInfo(self.robot, i, physicsClientId=self.cid)[2] != p.JOINT_FIXED
        ]
        infos = [p.getJointInfo(self.robot, j, physicsClientId=self.cid) for j in self.joint_indices]
        self.q_low = np.array([i[8] for i in infos])
        self.q_high = np.array([i[9] for i in infos])
        self.n = len(self.joint_indices)  # 7

        self.target_body = None
        if render_mode == "human":
            vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.04,
                                     rgbaColor=[1, 0, 0, 1], physicsClientId=self.cid)
            self.target_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs,
                                                 basePosition=[0, 0, 1], physicsClientId=self.cid)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3 * 3 + 2 * self.n + 1,),
                                            dtype=np.float32)

    # ---------- helpers ----------
    def set_difficulty(self, d):
        self.difficulty = float(np.clip(d, 0.0, 1.0))

    def _set_q(self, q):
        for j, qi in zip(self.joint_indices, q):
            p.resetJointState(self.robot, j, float(qi), targetVelocity=0.0,
                              physicsClientId=self.cid)

    def _ee(self):
        return np.array(p.getLinkState(self.robot, self.ee_index,
                                       computeForwardKinematics=True,
                                       physicsClientId=self.cid)[4])

    def _sample_q(self):
        q = self.HOME + self.difficulty * self.np_random.uniform(-1, 1, self.n) * self.SPREAD
        margin = 0.05 * (self.q_high - self.q_low)
        return np.clip(q, self.q_low + margin, self.q_high - margin)

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # reachable goal = FK of a random configuration
        for _ in range(50):
            self._set_q(self._sample_q())
            goal = self._ee()
            if goal[2] > 0.15:
                break
        self.target_pos = goal

        # start configuration, not trivially close to the goal
        for _ in range(50):
            q0 = self._sample_q()
            self._set_q(q0)
            if np.linalg.norm(self._ee() - self.target_pos) > 0.15:
                break
        self.q_target = q0.copy()
        self._apply_control()

        if self.target_body is not None:
            p.resetBasePositionAndOrientation(self.target_body, self.target_pos,
                                              [0, 0, 0, 1], physicsClientId=self.cid)

        self.prev_dist = np.linalg.norm(self._ee() - self.target_pos)
        return self._get_obs(), {}

    def _apply_control(self):
        p.setJointMotorControlArray(
            self.robot, self.joint_indices, p.POSITION_CONTROL,
            targetPositions=self.q_target,
            forces=[200.0] * self.n,
            physicsClientId=self.cid,
        )

    def step(self, action):
        self.current_step += 1
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.q_target = np.clip(self.q_target + a * self.action_scale, self.q_low, self.q_high)
        self._apply_control()
        for _ in range(self.sim_substeps):
            p.stepSimulation(physicsClientId=self.cid)

        ee = self._ee()
        dist = float(np.linalg.norm(ee - self.target_pos))

        reward = 20.0 * (self.prev_dist - dist) - 0.5 * dist - 0.005 * float(np.sum(a ** 2))
        self.prev_dist = dist

        terminated = bool(dist < self.success_radius)
        truncated = bool(self.current_step >= self.max_steps) and not terminated
        if terminated:
            reward += 10.0

        info = {"distance": dist, "is_success": terminated}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        js = p.getJointStates(self.robot, self.joint_indices, physicsClientId=self.cid)
        q = np.array([s[0] for s in js])
        qd = np.array([s[1] for s in js])
        ee = self._ee()
        rel = self.target_pos - ee
        return np.concatenate([q, qd, ee, self.target_pos, rel,
                               [np.linalg.norm(rel)]]).astype(np.float32)

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(self.cid)
