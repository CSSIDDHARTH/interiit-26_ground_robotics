import numpy as np
import pybullet as p
from gymnasium import spaces

from pick_place_env import PickAndPlaceEnv   # must be the improved version


class ObstacleEnv(PickAndPlaceEnv):
    """Pick-and-place with 2 pillar obstacles, built on the improved PickAndPlaceEnv.

    Added on top of pick-and-place:
      * pillars are placed near the straight peg->target line (so they really
        matter) but never on top of the peg, the target, or the arm's start pose
      * `difficulty` (curriculum) also controls the chance that each pillar is
        'active'; inactive pillars are parked far outside the workspace
      * observation includes obstacle positions, obstacle-relative vectors to the
        end effector and peg, and the live robot/peg clearance to each pillar
      * proximity penalty gives a gradient *before* contact; hitting a pillar
        ends the episode with a penalty (and is never a success)
    """
    OBS_HALF = np.array([0.02, 0.02, 0.1])
    N_OBS = 2
    PROX = 0.10          # repulsive field radius: penalty starts inside 10 cm
    K_REP = 1.0          # repulsive gain (raise if it still skims, lower if it hovers/times out)
    CAP = 0.15           # clearance reported in the observation is capped here

    def __init__(self, *args, obstacle_prob=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacle_prob = obstacle_prob   # None -> follow curriculum difficulty
        self.obstacle_ids = []
        for _ in range(self.N_OBS):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.OBS_HALF.tolist(),
                                         physicsClientId=self.cid)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=self.OBS_HALF.tolist(),
                                      rgbaColor=[0.2, 0.2, 0.2, 1], physicsClientId=self.cid)
            self.obstacle_ids.append(p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                basePosition=[2.0, 2.0, 0.1], physicsClientId=self.cid))
        self.obs_pos = np.tile(np.array([2.0, 2.0, 0.1]), (self.N_OBS, 1))
        self.clear = np.full(self.N_OBS, self.CAP)

        # parent obs (32) + obs pos (6) + obs-ee (6) + obs-peg (6) + clearance (2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(32 + 18 + self.N_OBS,),
                                            dtype=np.float32)

    # ---------- geometry helpers ----------
    def _box_dist(self, pt, c):
        d = np.maximum(np.abs(np.asarray(pt) - np.asarray(c)) - self.OBS_HALF, 0.0)
        return float(np.linalg.norm(d))

    def _refresh(self):
        p.performCollisionDetection(physicsClientId=self.cid)

    def _min_dist(self, body, obs_id):
        pts = p.getClosestPoints(body, obs_id, self.CAP, physicsClientId=self.cid)
        return min([c[8] for c in pts], default=self.CAP)

    def _clearances(self):
        return np.array([min(self._min_dist(self.robot, o), self._min_dist(self.peg_id, o))
                         for o in self.obstacle_ids])

    def _in_collision(self):
        self._refresh()
        for o in self.obstacle_ids:
            if p.getContactPoints(bodyA=self.robot, bodyB=o, physicsClientId=self.cid):
                return True
            if p.getContactPoints(bodyA=self.peg_id, bodyB=o, physicsClientId=self.cid):
                return True
        return False

    def _move_obstacles(self, positions):
        for o, pos in zip(self.obstacle_ids, positions):
            p.resetBasePositionAndOrientation(o, pos, [0, 0, 0, 1], physicsClientId=self.cid)
        self.obs_pos = np.array(positions, dtype=np.float64)

    def _far_positions(self):
        return [np.array([2.0 + k, 2.0, 0.1]) for k in range(self.N_OBS)]

    def _place_obstacles(self):
        peg, tgt = self.peg_start, self.target_pos
        for _ in range(60):
            cand, active = [], []
            for k in range(self.N_OBS):
                prob = self.difficulty if self.obstacle_prob is None else self.obstacle_prob
                if self.np_random.uniform() < prob:
                    t = self.np_random.uniform(0.25, 0.75)
                    c = peg + t * (tgt - peg) + self.np_random.uniform(-0.05, 0.05, 3)
                    c[2] = np.clip(c[2], 0.1, 0.5)
                    cand.append(c)
                    active.append(c)
                else:
                    cand.append(self._far_positions()[k])
            # keep peg and target OUTSIDE the repulsive field (PROX) so grasping and placing
            # are never penalised; the extra 0.03 covers the peg radius / success radius
            margin = self.PROX + 0.03
            ok = all(self._box_dist(peg, c) > margin and self._box_dist(tgt, c) > margin for c in active)
            if len(active) == 2 and np.linalg.norm(active[0] - active[1]) < 0.06:
                ok = False
            if not ok:
                continue
            self._move_obstacles(cand)
            self._refresh()
            if all(self._min_dist(self.robot, o) > 0.05 for o in self.obstacle_ids):
                return
        self._move_obstacles(self._far_positions())   # fallback: no obstacles this episode

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.peg_start = self._peg().copy()
        self._place_obstacles()
        self.prev_phi = self._phi()
        return self._get_obs(), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        collision = self._in_collision()

        # Repulsive potential field. self.clear[k] is the smallest distance between
        # pillar k and ANY robot link or the peg (from p.getClosestPoints), so the
        # whole arm is repelled, not just the end effector.
        # Penalty is 0 outside PROX and rises smoothly (quadratically) to K_REP at contact.
        d = np.maximum(self.clear, 0.0)
        prox = np.clip((self.PROX - d) / self.PROX, 0.0, 1.0)
        reward -= self.K_REP * float(np.sum(prox ** 2))

        if collision:
            if info["is_success"]:
                reward -= 50.0           # undo success bonus: collisions never count
            reward -= 10.0
            terminated, truncated = True, False
            info["is_success"] = False
        info["collision"] = collision
        info["min_clearance"] = float(self.clear.min())
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        base = super()._get_obs()
        self._refresh()
        self.clear = self._clearances()
        ee, peg = self._ee(), self._peg()
        extra = np.concatenate([
            self.obs_pos.ravel(),
            (self.obs_pos - ee).ravel(),
            (self.obs_pos - peg).ravel(),
            self.clear,
        ])
        return np.concatenate([base, extra]).astype(np.float32)
