import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import imageio
from envs.reach_env import UR5eReachEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .zip model")
    parser.add_argument("--algo", default="sac", choices=["ppo", "sac"])
    parser.add_argument("--episodes", type=int, default=4, help="Number of episodes to record")
    parser.add_argument("--out", default="results/demo.gif")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    if args.algo == "ppo":
        from stable_baselines3 import PPO as Algo
    else:
        from stable_baselines3 import SAC as Algo

    model = Algo.load(args.model)
    env = UR5eReachEnv(render_mode="rgb_array")

    frames = []
    successes = 0
    MIN_FRAMES_PER_EPISODE = 50  # keep recording briefly after success so motion is visible
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=100 + ep)  # seeds not used in training, for a "fresh" demo
        ep_frames = 0
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            frames.append(env.render())
            ep_frames += 1
            if trunc or (term and ep_frames >= MIN_FRAMES_PER_EPISODE):
                break
        if info["success"]:
            successes += 1
        print(f"Episode {ep+1}/{args.episodes}: success={info['success']}, dist={info['distance']:.3f}m")

    env.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"\nSaved {len(frames)} frames ({successes}/{args.episodes} episodes succeeded) to {args.out}")


if __name__ == "__main__":
    main()
