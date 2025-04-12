import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.distributions import Normal
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = gym.make('HalfCheetah-v5', render_mode=None)  # Use render_mode="human" for rendering

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

def estimate_advantages(states, next_states, rewards):
    rho=torch.mean(rewards)
    V_target=rewards-rho+critic(next_states)
    A_st_at=V_target-critic(states)
    return A_st_at, torch.mean((critic(states)-V_target)**2)

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

def conjugate_gradient(A, b, delta=0., max_iterations=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    for i in range(max_iterations):
        AVP = A(p)
        alpha = (r @ r) / (p @ AVP)
        x_new = x + alpha * p
        if (x - x_new).norm() <= delta:
            return x_new
        r = r - alpha * AVP
        beta = (r @ r) / (r @ r).clone()
        p = r + beta * p
        x = x_new
    return x

def apply_update(grad_flattened):
    n = 0
    for p in actor_params:
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel

def update_critic(critic_loss):
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

max_d_kl = 0.01

def update_agent(rollouts):
 
    states = torch.cat([r.states for r in rollouts], dim=0).to(device)
    next_states = torch.cat([r.next_states for r in rollouts], dim=0).to(device)
    rewards = torch.cat([r.rewards for r in rollouts], dim=0).to(device)
    actions = torch.cat([r.actions for r in rollouts], dim=0).to(device)

    advantages,critic_loss = estimate_advantages(states, next_states, rewards)
    advantages=advantages.flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    update_critic(critic_loss)
 
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
    eval_rewards=[]
    eval_interval=100000
    num_samples = 0
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
                num_samples += 1
                state = next_state


            states, actions, rewards, next_states = zip(*samples)
            states = torch.stack([torch.tensor(s, dtype=torch.float32, device=device) for s in states])
            next_states = torch.stack([torch.tensor(s, dtype=torch.float32, device=device) for s in next_states])
            actions = torch.stack([torch.tensor(a, dtype=torch.float32, device=device) for a in actions])
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states))
            rollout_total_rewards.append(rewards.sum().item())

        update_agent(rollouts)

        if num_samples // eval_interval > len(eval_rewards):
            avg_eval_reward = evaluate_policy(gym.make('HalfCheetah-v5'))  # Use new env to avoid interference
            eval_rewards.append(avg_eval_reward)
            print(f"Eval at {num_samples} samples: {avg_eval_reward:.2f}")

        mean_reward = np.mean(rollout_total_rewards)
        print(f'Epoch {epoch}, Mean Reward: {mean_reward:.2f}')
       

    print("Total samples collected:", num_samples)
    plt.plot(np.arange(1, len(eval_rewards)+1) * eval_interval, eval_rewards)
    plt.xlabel("Samples")
    plt.ylabel("Evaluation Reward")
    plt.title("Policy Evaluation over Time")
    plt.grid()
    plt.show()

# Train the agent
train(epochs=2)

env.close()

