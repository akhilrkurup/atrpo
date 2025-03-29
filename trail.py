import gymnasium as gym
import os
os.environ["MUJOCO_GL"] = "egl"

# Initialize the MuJoCo environment
env = gym.make("Walker2d-v5", render_mode="human",terminate_when_unhealthy=False)

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)

for _ in range(10000):
    # Sample a random action from the action space
    action = env.action_space.sample()

    # Step through the environment with the action
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended, reset the environment
    if terminated or truncated:
        observation, info = env.reset()

# Close the environment
env.close()