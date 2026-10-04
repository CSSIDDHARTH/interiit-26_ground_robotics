import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pick_place_env import PickAndPlaceEnv

N_EPISODES = 100
RENDER = len(sys.argv) > 1 and sys.argv[1] == "render"

env = DummyVecEnv([lambda: PickAndPlaceEnv(
    render_mode="human" if RENDER else None,
    difficulty=1.0, success_radius=0.05, p_start_grasped=0.0)])   # full task, no help
env = VecNormalize.load("vecnormalize_pick_place.pkl", env)
env.training = False
env.norm_reward = False

model = PPO.load("ppo_pick_place_model", env=env)

succ, grasp, dist = [], [], []
obs = env.reset()
while len(succ) < N_EPISODES:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, info = env.step(action)
    if done[0]:
        succ.append(bool(info[0]["is_success"]))
        grasp.append(bool(info[0]["grasped"]))
        dist.append(info[0]["peg_to_target"])

print(f"Success rate : {np.mean(succ):.2f}  ({sum(succ)}/{N_EPISODES})")
print(f"Grasp rate   : {np.mean(grasp):.2f}")
print(f"Mean final peg-to-target distance: {np.mean(dist):.3f} m")
