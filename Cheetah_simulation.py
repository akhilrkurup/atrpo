import argparse
import gymnasium as gym
import torch
import pickle
import numpy as np
import os
import time
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.torch_utils import tensor
from utils.tools import assets_dir


def simulate():
    parser = argparse.ArgumentParser(description="Simulate and visualize trained TRPO/ATRPO policies.")
    parser.add_argument("--env-name", type=str, default="HalfCheetah-v5",
                        help="Gymnasium MuJoCo environment name (e.g. HalfCheetah-v5, Ant-v5, Humanoid-v5)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to pre-trained model .p file (defaults to assets/learned_models/{env_name}_atrpo.p)")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max simulation steps (default: 1000)")
    parser.add_argument("--delay", type=float, default=0.01,
                        help="Delay per step in seconds for smoother rendering (default: 0.01)")
    args = parser.parse_args()

    # Determine model path
    if args.model_path is None:
        default_path = os.path.join(assets_dir(), "learned_models", f"{args.env_name}_atrpo.p")
        if not os.path.exists(default_path):
            default_path = os.path.join(assets_dir(), "learned_models", f"{args.env_name}_trpo_0.99.p")
        model_path = default_path
    else:
        model_path = args.model_path

    print(f"[+] Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    policy_net, value_net, running_state = pickle.load(open(model_path, "rb"))

    # Create environment with human rendering
    env = gym.make(args.env_name, render_mode="human")

    try:
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done and step_count < args.max_steps:
            if running_state is not None:
                obs = running_state(obs)

            obs_var = tensor(obs).unsqueeze(0)

            with torch.no_grad():
                action_mean, _, _ = policy_net(obs_var)
                action = action_mean.cpu().numpy()[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            if args.delay > 0:
                time.sleep(args.delay)

            total_reward += reward
            step_count += 1
            done = terminated or truncated

        print(f"[✔] Simulation complete! Total Steps: {step_count}, Total Undiscounted Reward: {total_reward:.2f}")

    finally:
        env.close()


if __name__ == "__main__":
    simulate()
