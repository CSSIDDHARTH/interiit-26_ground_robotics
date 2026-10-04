import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from pick_place_env import PickAndPlaceEnv

N_ENVS = 8
TOTAL_STEPS = 6_000_000


def make_env(rank):
    def _init():
        env = PickAndPlaceEnv(difficulty=0.3, success_radius=0.04,
                              p_start_grasped=0.25)   # 25% of episodes start holding the peg
        env = Monitor(env, info_keywords=("is_success",))
        env.reset(seed=rank)
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """Raise difficulty when recent success rate is high."""

    def __init__(self, check_every=2048, threshold=0.7, step=0.1):
        super().__init__()
        self.check_every, self.threshold, self.step_size = check_every, threshold, step
        self.difficulty = 0.3

    def _on_step(self):
        if self.n_calls % self.check_every == 0:
            buf = self.model.ep_info_buffer
            if len(buf) >= 100:
                sr = float(np.mean([e.get("is_success", 0.0) for e in buf]))
                self.logger.record("curriculum/success_rate", sr)
                if sr >= self.threshold and self.difficulty < 1.0:
                    self.difficulty = min(1.0, self.difficulty + self.step_size)
                    self.training_env.env_method("set_difficulty", self.difficulty)
                    buf.clear()
                    print(f"[curriculum] success {sr:.2f} -> difficulty {self.difficulty:.1f}")
            self.logger.record("curriculum/difficulty", self.difficulty)
        return True


if __name__ == "__main__":
    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy", venv,
        learning_rate=lambda progress_remaining: 3e-4 * progress_remaining,
        n_steps=512,            # x 8 envs = 4096 samples per update
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,         # a little exploration for the gripper dimension
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.7),
        tensorboard_log="./ppo_pick_place_tensorboard/",
        verbose=1,
    )

    callbacks = [
        CurriculumCallback(),
        CheckpointCallback(save_freq=250_000 // N_ENVS, save_path="./checkpoints_pp/",
                           name_prefix="ppo_pick_place", save_vecnormalize=True),
    ]
    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks)

    model.save("ppo_pick_place_model")
    venv.save("vecnormalize_pick_place.pkl")   # REQUIRED for testing
    print("Saved ppo_pick_place_model.zip and vecnormalize_pick_place.pkl")
