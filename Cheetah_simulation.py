import gymnasium as gym
import torch
import pickle
import numpy as np
import os
import time
from utils.torch_utils import *
# Load model
model_path = "/home/ila/Desktop/RL Project/atrpo_new/assets/learned_models/HalfCheetah-v5_trpo_0.99.p"
policy_net, value_net, running_state = pickle.load(open(model_path, "rb"))

# Create environment with rendering
env = gym.make("HalfCheetah-v5", render_mode="human")

try:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if running_state is not None:
            obs = running_state(obs)
         
        #obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_var = tensor(obs).unsqueeze(0)
    
        with torch.no_grad():
            action_mean, _, _ = policy_net(obs_var)
            action = action_mean.cpu().numpy()[0]
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        time.sleep(0.2)  # Slow down rendering

        total_reward += reward
        done = terminated or truncated

    print(f"Total reward: {total_reward:.2f}")

finally:
    env.close()
