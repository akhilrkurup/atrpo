import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Normal
from collections import namedtuple

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
env = gym.make("Walker2d-v5")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

gamma = 0.99
max_d_kl = 0.01
Rollout = namedtuple("Rollout", ["states", "actions", "rewards", "next_states"])

# Actor network
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.net(x)

# Critic network
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

actor = Actor().to(device)
critic = Critic().to(device)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

def get_action(state):
    state = torch.tensor(state, dtype=torch.float32, device=device)
    with torch.no_grad():
        mean = actor(state)
        dist = Normal(mean, 0.2)
        action = dist.sample()
    return action.cpu().numpy()

def estimate_advantages(states, last_state, rewards):
    with torch.no_grad():
        values = critic(states)
        next_value = critic(last_state.unsqueeze(0))
        values = torch.cat((values, next_value), dim=0)

    advantages = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = delta + gamma * advantages[t + 1] if t + 1 < len(rewards) else delta
    return advantages

def surrogate_loss(new_probs, old_probs, advantages):
    return (new_probs / old_probs * advantages).mean()

def kl_div(old_dist, new_dist):
    return torch.distributions.kl_divergence(old_dist, new_dist).mean()

def flat_grad(y, model_params, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(y, model_params, retain_graph=retain_graph, create_graph=create_graph)
    return torch.cat([g.view(-1) for g in grads])

def conjugate_gradient(HVP, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Avp = HVP(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x

def update_agent(rollouts):
    states = torch.cat([r.states for r in rollouts], dim=0).to(device)
    actions = torch.cat([r.actions for r in rollouts], dim=0).to(device)
    advantages = [estimate_advantages(r.states.to(device), r.next_states[-1].to(device), r.rewards.to(device)) for r in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten()
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    std = advantages.std()
    std = std if std > 1e-6 else 1.0
    advantages = (advantages - advantages.mean()) / std

    critic_optimizer.zero_grad()
    critic_loss = 0.5 * (advantages ** 2).mean()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
    critic_optimizer.step()

    dist_old = Normal(actor(states), 0.2)
    old_log_probs = dist_old.log_prob(actions).sum(-1).detach()
    dist = Normal(actor(states), 0.2)
    probabilities = dist.log_prob(actions).sum(-1).exp()
    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(dist, dist)

    parameters = list(actor.parameters())
    g = flat_grad(L, parameters, retain_graph=True)
    d_kl = flat_grad(KL, parameters, create_graph=True)

    def HVP(v):
        return flat_grad(d_kl @ v, parameters, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g)
    max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir
    old_params = [p.clone() for p in actor.parameters()]

    def apply_update(step):
        n = 0
        for p in actor.parameters():
            numel = p.numel()
            p.data += step[n:n + numel].view(p.shape)
            n += numel

    def criterion(step):
        apply_update(step)
        new_mean = actor(states)
        if torch.isnan(new_mean).any():
            print("❌ NaNs detected in actor output during update. Reverting step.")
            apply_update(-step)
            return False
        new_dist = Normal(new_mean, 0.2)
        new_log_probs = new_dist.log_prob(actions).sum(-1).exp()
        L_new = surrogate_loss(new_log_probs, probabilities, advantages)
        KL_new = kl_div(dist, new_dist)
        if L_new - L > 0 and KL_new <= max_d_kl:
            return True
        apply_update(-step)
        return False

    i = 0
    while not criterion((0.9 ** i) * max_step) and i < 10:
        i += 1

def train(epochs=100, num_rollouts=50):
    mean_total_rewards = []
    for epoch in range(epochs):
        rollouts = []
        rollout_total_rewards = []
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
            states, actions, rewards, next_states = zip(*samples)
            rewards_np = np.array(rewards)
            print("🎯 Reward Stats → Max:", rewards_np.max(), "Min:", rewards_np.min(), "Mean:", rewards_np.mean())
            states = torch.tensor(np.array(states), dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
            rollouts.append(Rollout(states, actions, rewards, next_states))
            rollout_total_rewards.append(rewards.sum().item())
        update_agent(rollouts)
        mean_reward = np.mean(rollout_total_rewards)
        print(f'✅ Epoch {epoch}, Mean Reward: {mean_reward}')
        mean_total_rewards.append(mean_reward)
    plt.plot(mean_total_rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Reward")
    plt.title("TRPO Training Progress")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    train()
