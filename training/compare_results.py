import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from envs.reach_env import UR5eReachEnv
from stable_baselines3 import PPO, SAC

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_history(name):
    path = RESULTS_DIR / f"{name}_history.json"
    if not path.exists():
        print(f"Warning: {path} not found. Run train_{name}.py first.")
        return None
    with open(path) as f:
        return json.load(f)


def plot_learning_curves():
    ppo_hist = load_history("ppo")
    sac_hist = load_history("sac")

    plt.figure(figsize=(8, 5))
    if ppo_hist:
        steps = [h["timestep"] for h in ppo_hist]
        rewards = [h["mean_reward"] for h in ppo_hist]
        plt.plot(steps, rewards, label="PPO", linewidth=2)
    if sac_hist:
        steps = [h["timestep"] for h in sac_hist]
        rewards = [h["mean_reward"] for h in sac_hist]
        plt.plot(steps, rewards, label="SAC", linewidth=2)

    plt.xlabel("Training timesteps")
    plt.ylabel("Mean episode reward")
    plt.title("PPO vs SAC: UR5e Reach Task Learning Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = RESULTS_DIR / "learning_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


def evaluate_policy(model, n_episodes=20):
    """Run fixed evaluation episodes and return success rate + mean final distance."""
    env = UR5eReachEnv()
    successes = 0
    final_dists = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)  # fixed seeds -> same targets across algorithms
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break
        final_dists.append(info["distance"])
        if info["success"]:
            successes += 1
    return successes / n_episodes, np.mean(final_dists)


def main():
    plot_learning_curves()

    summary = {}
    ppo_path = RESULTS_DIR / "ppo_reach_model.zip"
    sac_path = RESULTS_DIR / "sac_reach_model.zip"

    if ppo_path.exists():
        model = PPO.load(str(ppo_path))
        success_rate, mean_dist = evaluate_policy(model)
        summary["PPO"] = {"success_rate": success_rate, "mean_final_distance_m": round(mean_dist, 4)}

    if sac_path.exists():
        model = SAC.load(str(sac_path))
        success_rate, mean_dist = evaluate_policy(model)
        summary["SAC"] = {"success_rate": success_rate, "mean_final_distance_m": round(mean_dist, 4)}

    print("\n=== Evaluation Summary (20 fixed episodes each) ===")
    for algo, stats in summary.items():
        print(f"{algo}: success_rate={stats['success_rate']*100:.0f}%  "
              f"mean_final_distance={stats['mean_final_distance_m']}m")

    with open(RESULTS_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {RESULTS_DIR / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
