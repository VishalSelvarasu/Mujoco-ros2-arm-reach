import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.reach_env import UR5eReachEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import json

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class EpisodeLogger(BaseCallback):
    """Logs mean episode reward + success rate periodically for comparison plots."""

    def __init__(self, log_every=2000, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every
        self.history = []

    def _on_step(self):
        if self.n_calls % self.log_every == 0:
            ep_info_buffer = self.model.ep_info_buffer
            if len(ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in ep_info_buffer])
                mean_len = np.mean([ep["l"] for ep in ep_info_buffer])
                self.history.append({
                    "timestep": self.num_timesteps,
                    "mean_reward": float(mean_reward),
                    "mean_ep_length": float(mean_len),
                })
                print(f"[PPO] step {self.num_timesteps} | mean_reward {mean_reward:.2f}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n_envs", type=int, default=8)
    args = parser.parse_args()

    env = make_vec_env(UR5eReachEnv, n_envs=args.n_envs)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log=str(RESULTS_DIR / "tb_ppo"),
    )

    logger_cb = EpisodeLogger(log_every=2000)
    model.learn(total_timesteps=args.timesteps, callback=logger_cb)

    model.save(str(RESULTS_DIR / "ppo_reach_model"))
    with open(RESULTS_DIR / "ppo_history.json", "w") as f:
        json.dump(logger_cb.history, f, indent=2)

    print(f"Saved model to {RESULTS_DIR / 'ppo_reach_model.zip'}")
    print(f"Saved training history to {RESULTS_DIR / 'ppo_history.json'}")


if __name__ == "__main__":
    main()
