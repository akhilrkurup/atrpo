import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.distributions import Normal
from collections import namedtuple
import time

start = time.time()


env = gym.make('HalfCheetah-v5', render_mode=None)  # Change to "human" for visualization

state_size = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]  # Continuous action space

Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states'])

def train(epochs=100, num_rollouts=50):
    mean_total_rewards = []
    global_rollout = 0
    num_samples=0
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
                num_samples+=1
                state = next_state
            
            states, actions, rewards, next_states = zip(*samples)
            states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
            next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_states])
            actions = torch.stack([torch.tensor(a, dtype=torch.float32) for a in actions])
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            
            rollouts.append(Rollout(states, actions, rewards, next_states))
            rollout_total_rewards.append(rewards.sum().item())
            global_rollout += 1
        
        update_agent(rollouts)
        mean_reward = np.mean(rollout_total_rewards)
        print(f'Epoch {epoch}, Mean Reward: {mean_reward}')
        mean_total_rewards.append(mean_reward)
    print("Total samples collected:", num_samples)
    plt.plot(np.linspace(0,num_samples,epochs),mean_total_rewards)
    plt.show()
    

# Actor network (continuous output)
actor_hidden = 64
actor = nn.Sequential(
    nn.Linear(state_size, actor_hidden),
    nn.Tanh(),
    nn.Linear(actor_hidden, actor_hidden),
    nn.Tanh(),
    nn.Linear(actor_hidden, num_actions)
)
log_std = nn.Parameter(torch.full((num_actions,), -0.5))
actor_params=list(actor.parameters()) + [log_std]

def get_action(state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    mean = actor(state)
    std = log_std.exp().expand_as(mean)
    dist = Normal(mean, std)
    action = dist.sample()
    return action.squeeze(0).numpy()

critic_hidden = 64
critic = nn.Sequential(
    nn.Linear(state_size, critic_hidden),
    nn.Tanh(),
    nn.Linear(critic_hidden, critic_hidden),
    nn.Tanh(),
    nn.Linear(critic_hidden, 1)
)
critic_optimizer = Adam(critic.parameters(), lr=3e-4, weight_decay=3e-3)

def update_critic(advantages):
    loss = 0.5 * (advantages ** 2).mean()
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()

max_d_kl = 0.01

def update_agent(rollouts):
    states = torch.cat([r.states for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0)
    
    advantages = [estimate_advantages(r.states, r.next_states[-1], r.rewards) for r in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten()
    print("adv shape",advantages.shape)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    update_critic(advantages)
    
    dist = Normal(actor(states), torch.exp(log_std).expand_as(actor(states)))
    probabilities = dist.log_prob(actions).sum(dim=-1).exp()
    
    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(dist, dist)
    
    parameters = actor_params
    g = flat_grad(L, parameters, retain_graph=True)
    d_kl = flat_grad(KL, parameters, create_graph=True)


    def HVP(v):
        return flat_grad(d_kl @ v, parameters, retain_graph=True)
    
    search_dir = conjugate_gradient(HVP, g)
    max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir
    
    def criterion(step):
        apply_update(step)
        dist_new = Normal(actor(states), torch.exp(log_std).expand_as(actor(states)))
        probabilities_new = dist_new.log_prob(actions).sum(dim=-1).exp()
        L_new = surrogate_loss(probabilities_new, probabilities, advantages)
        KL_new = kl_div(dist, dist_new)
        if L_new - L > 0 and KL_new <= max_d_kl:
            return True
        apply_update(-step)
        return False
    
    i = 0
    while not criterion((0.9 ** i) * max_step) and i < 10:
        i += 1

def estimate_advantages(states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
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

train(epochs=100, num_rollouts=10)
env.close()
end = time.time()
print(f"Elapsed time: {end - start:.3f} seconds")
env=gym.make('HalfCheetah-v5', render_mode="human")
state, _ = env.reset()
cum_reward = 0
done = False
while not done:
    action = get_action(state)
    state, reward, terminated, truncated, _ = env.step(action)
    cum_reward+=reward
    done = terminated or truncated
print("Total reward:", cum_reward)
env.close()

import time

start = time.time()
# your code here
