import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np


class PickAndPlaceEnv(gym.Env):
    """KUKA iiwa pick-and-place with a constraint-based (magnetic) gripper.

    Improvements over the first version:
      * normalized [-1, 1] actions, commanded joint targets (no sag drift)
      * peg / target positions come from forward kinematics -> always reachable
      * peg rests "pinned" in place until grasped (and re-pinned if dropped),
        so a bad release never makes the task unrecoverable
      * grasp constraint preserves the current relative pose (no teleport jolt)
      * peg-robot collisions disabled (abstract gripper)
      * gripper hysteresis (+-0.3 dead-band) so noisy actions don't chatter
      * unified potential reward  phi = -(|ee-peg| + |peg-target|)
        + one-time grasp bonus + release-at-target bonus + success bonus
      * `p_start_grasped`: some training episodes start already holding the
        peg (stage-2 practice), which massively speeds up learning
      * difficulty knob for curriculum
    """
    metadata = {"render_modes": ["human"]}

    HOME = np.array([0.0, 0.4, 0.0, -1.2, 0.0, 0.8, 0.0])
    SPREAD = np.array([1.2, 0.7, 1.2, 0.7, 1.2, 0.7, 1.2])

    def __init__(self, render_mode=None, max_steps=250, action_scale=0.05,
                 sim_substeps=10, success_radius=0.05, grasp_radius=0.06,
                 difficulty=1.0, p_start_grasped=0.0, auto_release=True):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.sim_substeps = sim_substeps
        self.success_radius = success_radius
        self.grasp_radius = grasp_radius
        self.difficulty = float(difficulty)
        self.p_start_grasped = p_start_grasped
        # True: the gripper opens automatically once the peg is inside the target radius,
        # so the policy only has to learn "carry it there" (explicit release was the bottleneck).
        # False: the policy must also output grip < -0.3 at the target.
        self.auto_release = auto_release

        self.cid = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True,
                                physicsClientId=self.cid)
        self.ee_index = 6
        self.joint_indices = [
            i for i in range(p.getNumJoints(self.robot, physicsClientId=self.cid))
            if p.getJointInfo(self.robot, i, physicsClientId=self.cid)[2] != p.JOINT_FIXED
        ]
        infos = [p.getJointInfo(self.robot, j, physicsClientId=self.cid) for j in self.joint_indices]
        self.q_low = np.array([i[8] for i in infos])
        self.q_high = np.array([i[9] for i in infos])
        self.n = len(self.joint_indices)

        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.1, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.1,
                                  rgbaColor=[0, 0, 1, 1], physicsClientId=self.cid)
        self.peg_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis, basePosition=[0.4, 0, 0.3],
                                        physicsClientId=self.cid)
        for link in range(-1, p.getNumJoints(self.robot, physicsClientId=self.cid)):
            p.setCollisionFilterPair(self.robot, self.peg_id, link, -1, 0, physicsClientId=self.cid)

        self.target_id = None
        if render_mode == "human":
            vt = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 0.5],
                                     physicsClientId=self.cid)
            self.target_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vt,
                                               basePosition=[0, 0, 1], physicsClientId=self.cid)

        self.constraint = None   # either a world pin or the grasp constraint
        self.holding = False

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n + 1,), dtype=np.float32)
        # q(7) qd(7) ee(3) peg(3) target(3) peg-ee(3) target-peg(3) |peg-ee| |tgt-peg| holding = 32
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2 * self.n + 18,),
                                            dtype=np.float32)

    # ---------- helpers ----------
    def set_difficulty(self, d):
        self.difficulty = float(np.clip(d, 0.0, 1.0))

    def _set_q(self, q):
        for j, qi in zip(self.joint_indices, q):
            p.resetJointState(self.robot, j, float(qi), targetVelocity=0.0, physicsClientId=self.cid)

    def _ee(self):
        return np.array(p.getLinkState(self.robot, self.ee_index, computeForwardKinematics=True,
                                       physicsClientId=self.cid)[4])

    def _peg(self):
        return np.array(p.getBasePositionAndOrientation(self.peg_id, physicsClientId=self.cid)[0])

    def _sample_q(self):
        q = self.HOME + self.difficulty * self.np_random.uniform(-1, 1, self.n) * self.SPREAD
        m = 0.05 * (self.q_high - self.q_low)
        return np.clip(q, self.q_low + m, self.q_high - m)

    def _sample_point(self, zmin=0.12, zmax=0.40):
        for _ in range(100):
            self._set_q(self._sample_q())
            pt = self._ee()
            if zmin < pt[2] < zmax and pt[0] > 0.2:
                return pt
        return np.array([0.4, 0.0, 0.25])

    def _remove_constraint(self):
        if self.constraint is not None:
            p.removeConstraint(self.constraint, physicsClientId=self.cid)
            self.constraint = None

    def _pin_peg(self):
        """Freeze the peg in place (it 'rests' on something)."""
        self._remove_constraint()
        pos, orn = p.getBasePositionAndOrientation(self.peg_id, physicsClientId=self.cid)
        p.resetBaseVelocity(self.peg_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        self.constraint = p.createConstraint(
            self.peg_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos,
            childFrameOrientation=orn, physicsClientId=self.cid)
        self.holding = False

    def _grasp_peg(self):
        """Attach peg to the end effector, preserving the current relative pose."""
        self._remove_constraint()
        ls = p.getLinkState(self.robot, self.ee_index, computeForwardKinematics=True,
                            physicsClientId=self.cid)
        com_pos, com_orn = ls[0], ls[1]
        peg_pos, peg_orn = p.getBasePositionAndOrientation(self.peg_id, physicsClientId=self.cid)
        inv_pos, inv_orn = p.invertTransform(com_pos, com_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, peg_pos, peg_orn)
        self.constraint = p.createConstraint(
            self.robot, self.ee_index, self.peg_id, -1, p.JOINT_FIXED, [0, 0, 0],
            rel_pos, [0, 0, 0], parentFrameOrientation=rel_orn, physicsClientId=self.cid)
        self.holding = True

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._remove_constraint()
        self.holding = False
        self.grasped_once = False

        peg_pos = self._sample_point()
        for _ in range(50):
            target = self._sample_point()
            if np.linalg.norm(target - peg_pos) > 0.2:
                break
        self.target_pos = target

        p.resetBasePositionAndOrientation(self.peg_id, peg_pos, [0, 0, 0, 1], physicsClientId=self.cid)
        p.resetBaseVelocity(self.peg_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

        started_grasped = False
        if self.np_random.uniform() < self.p_start_grasped:
            jitter = self.np_random.uniform(-0.02, 0.02, 3)
            q_ik = np.array(p.calculateInverseKinematics(
                self.robot, self.ee_index, (peg_pos + jitter).tolist(),
                lowerLimits=self.q_low.tolist(), upperLimits=self.q_high.tolist(),
                jointRanges=(self.q_high - self.q_low).tolist(), restPoses=self.HOME.tolist(),
                maxNumIterations=100, physicsClientId=self.cid))[: self.n]
            self._set_q(q_ik)
            if np.linalg.norm(self._ee() - peg_pos) < 0.03:
                started_grasped = True
                self.q_target = q_ik.copy()
        if not started_grasped:
            for _ in range(50):
                q0 = self._sample_q()
                self._set_q(q0)
                if np.linalg.norm(self._ee() - peg_pos) > 0.15:
                    break
            self.q_target = q0.copy()

        self._apply_control()
        if started_grasped:
            self._grasp_peg()
            self.grasped_once = True
        else:
            self._pin_peg()

        if self.target_id is not None:
            p.resetBasePositionAndOrientation(self.target_id, self.target_pos, [0, 0, 0, 1],
                                              physicsClientId=self.cid)
        self.prev_phi = self._phi()
        return self._get_obs(), {}

    def _apply_control(self):
        p.setJointMotorControlArray(self.robot, self.joint_indices, p.POSITION_CONTROL,
                                    targetPositions=self.q_target, forces=[200.0] * self.n,
                                    physicsClientId=self.cid)

    def _phi(self):
        return -(np.linalg.norm(self._ee() - self._peg()) + np.linalg.norm(self._peg() - self.target_pos))

    def step(self, action):
        self.current_step += 1
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        arm, grip = a[: self.n], a[self.n]

        self.q_target = np.clip(self.q_target + arm * self.action_scale, self.q_low, self.q_high)
        self._apply_control()

        # ----- gripper (hysteresis dead-band) -----
        just_grasped = just_released = False
        d_ep = np.linalg.norm(self._ee() - self._peg())
        d_pt0 = float(np.linalg.norm(self._peg() - self.target_pos))
        near_unheld = (not self.holding) and d_ep < self.grasp_radius
        if not self.holding and grip > 0.3 and d_ep < self.grasp_radius:
            self._grasp_peg()
            just_grasped = True
        elif self.holding and ((not self.auto_release and grip < -0.3) or
                               (self.auto_release and d_pt0 < 0.8 * self.success_radius)):
            self._pin_peg()          # frozen where it is -> can be re-grasped
            just_released = True

        for _ in range(self.sim_substeps):
            p.stepSimulation(physicsClientId=self.cid)

        peg = self._peg()
        d_pt = float(np.linalg.norm(peg - self.target_pos))

        # ----- reward -----
        phi = self._phi()
        reward = 20.0 * (phi - self.prev_phi) - 0.005 * float(np.sum(arm ** 2))
        self.prev_phi = phi

        # hovering at the peg without closing the gripper was a common failure -> nudge "grip"
        if near_unheld and not just_grasped:
            reward += 0.2 * grip

        if just_grasped and not self.grasped_once:
            reward += 5.0
            self.grasped_once = True
        if self.holding and d_pt < self.success_radius:
            reward += 0.5 * (-grip)             # encourage releasing at the target

        terminated = bool(just_released and d_pt < self.success_radius)
        if terminated:
            reward += 50.0
        truncated = bool(self.current_step >= self.max_steps) and not terminated

        info = {"is_success": terminated, "peg_to_target": d_pt, "grasped": self.grasped_once}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        js = p.getJointStates(self.robot, self.joint_indices, physicsClientId=self.cid)
        q = np.array([s[0] for s in js])
        qd = np.array([s[1] for s in js])
        ee, peg = self._ee(), self._peg()
        return np.concatenate([
            q, qd, ee, peg, self.target_pos, peg - ee, self.target_pos - peg,
            [np.linalg.norm(peg - ee), np.linalg.norm(self.target_pos - peg), float(self.holding)],
        ]).astype(np.float32)

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(self.cid)
