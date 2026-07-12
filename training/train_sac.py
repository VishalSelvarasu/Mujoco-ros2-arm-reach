import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.reach_env import UR5eReachEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import json

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class EpisodeLogger(BaseCallback):
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
                print(f"[SAC] step {self.num_timesteps} | mean_reward {mean_reward:.2f}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    args = parser.parse_args()

    # SAC is off-policy, so unlike PPO we use a single env (no need for
    # many parallel envs to fill the replay buffer effectively).
    env = UR5eReachEnv()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        tensorboard_log=str(RESULTS_DIR / "tb_sac"),
    )

    logger_cb = EpisodeLogger(log_every=2000)
    model.learn(total_timesteps=args.timesteps, callback=logger_cb)

    model.save(str(RESULTS_DIR / "sac_reach_model"))
    with open(RESULTS_DIR / "sac_history.json", "w") as f:
        json.dump(logger_cb.history, f, indent=2)

    print(f"Saved model to {RESULTS_DIR / 'sac_reach_model.zip'}")
    print(f"Saved training history to {RESULTS_DIR / 'sac_history.json'}")


if __name__ == "__main__":
    main()
