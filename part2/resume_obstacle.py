"""Resume obstacle training from a saved checkpoint (new 52-dim env only).

Usage:
  python resume_obstacle.py <model.zip> <vecnormalize.pkl> <start_difficulty> [extra_steps]

Example (files written by CheckpointCallback or by train_obstacle.py):
  python resume_obstacle.py ppo_obstacle_model.zip vecnormalize_obstacle.pkl 0.6 5000000
"""
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from obstacle_env import ObstacleEnv
from train_obstacle import CurriculumCallback, N_ENVS

model_path, vec_path = sys.argv[1], sys.argv[2]
start_diff = float(sys.argv[3])
extra_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 5_000_000


def make_env(rank):
    def _init():
        env = ObstacleEnv(difficulty=start_diff, success_radius=0.04, p_start_grasped=0.25)
        env = Monitor(env, info_keywords=("is_success", "collision"))
        env.reset(seed=1000 + rank)
        return env
    return _init


if __name__ == "__main__":
    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    venv = VecNormalize.load(vec_path, venv)   # keep the learned observation statistics
    venv.training = True
    venv.norm_reward = True

    # constant, smaller learning rate for fine-tuning
    model = PPO.load(model_path, env=venv,
                     custom_objects={"learning_rate": 1e-4, "lr_schedule": lambda _: 1e-4})

    cb = CurriculumCallback()
    cb.difficulty = start_diff
    ckpt = CheckpointCallback(save_freq=500_000 // N_ENVS, save_path="./checkpoints_obs/",
                              name_prefix="ppo_obstacle_resume", save_vecnormalize=True)

    model.learn(total_timesteps=extra_steps, callback=[cb, ckpt], reset_num_timesteps=False)
    model.save("ppo_obstacle_model")
    venv.save("vecnormalize_obstacle.pkl")
    print("Saved ppo_obstacle_model.zip and vecnormalize_obstacle.pkl")
