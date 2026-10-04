import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from manipulator_env import ManipulatorReachEnv

N_EPISODES = 100
RENDER = len(sys.argv) > 1 and sys.argv[1] == "render"

env = DummyVecEnv([lambda: ManipulatorReachEnv(
    render_mode="human" if RENDER else None,
    difficulty=1.0, success_radius=0.05)])
env = VecNormalize.load("vecnormalize.pkl", env)
env.training = False        # freeze running statistics
env.norm_reward = False

model = PPO.load("ppo_reach_model", env=env)

successes, final_dists = [], []
obs = env.reset()
while len(successes) < N_EPISODES:
    action, _ = model.predict(obs, deterministic=True)   # deterministic!
    obs, reward, done, info = env.step(action)
    if done[0]:
        successes.append(bool(info[0]["is_success"]))
        final_dists.append(info[0]["distance"])

print(f"Success rate : {np.mean(successes):.2f}  ({sum(successes)}/{N_EPISODES})")
print(f"Mean final distance: {np.mean(final_dists):.3f} m")
