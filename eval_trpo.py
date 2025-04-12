import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.distributions import Normal
from collections import namedtuple
import argparse
import os


os.makedirs("logs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True, help="Gymnasium environment name (e.g., HalfCheetah-v5)")
args = parser.parse_args()

env_name = args.env
plot_file = f"logs/{env_name}_eval_plot.png"

env = gym.make(env_name, render_mode=None)  # Use render_mode="human" for rendering

state_size = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]  # Continuous action space

Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states'])

actor_hidden = 64
actor = nn.Sequential(
    nn.Linear(state_size, actor_hidden),
    nn.Tanh(),
    nn.Linear(actor_hidden, actor_hidden),
    nn.Tanh(),
    nn.Linear(actor_hidden, num_actions)
).to(device)

log_std = nn.Parameter(torch.full((num_actions,), -0.5, device=device))
actor_params = list(actor.parameters()) + [log_std]

critic_hidden = 64
critic = nn.Sequential(
    nn.Linear(state_size, critic_hidden),
    nn.Tanh(),
    nn.Linear(critic_hidden, critic_hidden),
    nn.Tanh(),
    nn.Linear(critic_hidden, 1)
).to(device)

critic_optimizer = Adam(critic.parameters(), lr=3e-4, weight_decay=3e-3)

def get_action(state):
    if isinstance(state, np.ndarray):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        state = state.unsqueeze(0).to(device)
    mean = actor(state)
    std = log_std.exp().expand_as(mean)
    dist = Normal(mean, std)
    action = dist.sample()
    return action.squeeze(0).cpu().numpy()  # return to CPU for env.step()

def estimate_advantages(states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0).to(device))
    next_values = torch.zeros_like(rewards)
    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + 0.99 * last_value
    return next_values - values

def surrogate_loss(new_log_probs, old_log_probs, advantages):
    ratios = (new_log_probs - old_log_probs).exp()
    return (ratios * advantages).mean()

def kl_div(p, q):
    mean_p, std_p = p.mean.detach(), p.stddev.detach()
    mean_q, std_q = q.mean, q.stddev
    return (torch.log(std_q / std_p) + (std_p**2 + (mean_p - mean_q)**2) / (2.0 * std_q**2) - 0.5).sum(-1).mean()

def flat_grad(y, x, retain_graph=True, create_graph=False):
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    return torch.cat([t.view(-1) for t in g])

def conjugate_gradient(Avp_fn, b, delta=1e-10, max_iterations=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    r_dot_old = r @ r

    for _ in range(max_iterations):
        Avp = Avp_fn(p)  # This uses autograd to compute A·p
        alpha = r_dot_old / (p @ Avp)
        x_new = x + alpha * p

        if (x_new - x).norm() <= delta:
            break

        r = r - alpha * Avp
        r_dot_new = r @ r
        beta = r_dot_new / r_dot_old

        p = r + beta * p
        r_dot_old = r_dot_new
        x = x_new

    return x

def apply_update(grad_flattened):
    n = 0
    for p in actor_params:
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel

def update_critic(advantages):
    loss = 0.5 * (advantages ** 2).mean()
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()

max_d_kl = 0.01

def update_agent(rollouts):
    states = torch.cat([r.states for r in rollouts], dim=0).to(device)
    actions = torch.cat([r.actions for r in rollouts], dim=0).to(device)
    
    advantages = [estimate_advantages(r.states.to(device), r.next_states[-1].to(device), r.rewards.to(device)) for r in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    update_critic(advantages)

    dist = Normal(actor(states), log_std.exp().expand_as(actor(states)))
    probabilities = dist.log_prob(actions).sum(dim=-1)

    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(dist, dist)

    g = flat_grad(L, actor_params, retain_graph=True)
    d_kl = flat_grad(KL, actor_params, create_graph=True)

    def HVP(v):
        return flat_grad(d_kl @ v, actor_params, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g)
    max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir

    def criterion(step):
        apply_update(step)
        dist_new = Normal(actor(states), log_std.exp().expand_as(actor(states)))
        probabilities_new = dist_new.log_prob(actions).sum(dim=-1)
        L_new = surrogate_loss(probabilities_new, probabilities, advantages)
        KL_new = kl_div(dist, dist_new)
        if L_new - L > 0 and KL_new <= max_d_kl:
            return True
        apply_update(-step)
        return False

    i = 0
    while not criterion((0.9 ** i) * max_step) and i < 10:
        i += 1
def evaluate_policy(env, eval_runs=10):
    total_rewards = []
    for _ in range(eval_runs):
        state, _ = env.reset()
        done = False
        cum_reward = 0
        while not done:
            action = get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            cum_reward += reward
            done = terminated or truncated
        total_rewards.append(cum_reward)
    return np.mean(total_rewards)

def train(epochs=100, num_rollouts=5):
    eval_rewards = []
    eval_interval = 100000
    total_samples = 0  # total samples across all epochs

    for epoch in range(epochs):
        rollouts = []
        rollout_total_rewards = []
        epoch_samples = 0  # samples within current epoch

        for _ in range(num_rollouts):
            state, _ = env.reset()
            done = False
            samples = []

            while not done:
                action = get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                samples.append((state, action, reward, next_state))
                state = next_state

            # convert samples to tensors
            states, actions, rewards, next_states = zip(*samples)
            states = torch.stack([torch.tensor(s, dtype=torch.float32, device=device) for s in states])
            next_states = torch.stack([torch.tensor(s, dtype=torch.float32, device=device) for s in next_states])
            actions = torch.stack([torch.tensor(a, dtype=torch.float32, device=device) for a in actions])
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states))
            rollout_total_rewards.append(rewards.sum().item())

            num_samples = len(samples)
            total_samples += num_samples
            epoch_samples += num_samples

            if epoch_samples >= 5000:
                print(f"Epoch {epoch}: Reached 5000 samples, ending early with {epoch_samples} samples.")
                break  # early end of epoch

        update_agent(rollouts)

        if total_samples // eval_interval > len(eval_rewards):
            avg_eval_reward = evaluate_policy(gym.make(env_name))  # Use a fresh env
            eval_rewards.append(avg_eval_reward)
            print(f"Eval at {total_samples} samples: {avg_eval_reward:.2f}")

        mean_reward = np.mean(rollout_total_rewards)
        print(f'Epoch {epoch}, Mean Reward: {mean_reward:.2f}')

       

    print("Total samples collected:", total_samples)
    plt.plot(np.arange(1, len(eval_rewards)+1) * eval_interval, eval_rewards)
    plt.xlabel("Samples")
    plt.ylabel("Evaluation Reward")
    plt.title("Policy Evaluation over Time")
    plt.grid()
    plt.savefig(plot_file)

# Train the agent
train(epochs=25)

env.close()

