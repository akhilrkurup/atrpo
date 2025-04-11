import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.distributions import Normal
from collections import namedtuple


class ResetCostWrapper(gym.Wrapper):
    def __init__(self, env, reset_cost=100.0):
        super().__init__(env)
        self.reset_cost = reset_cost

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if 'reset_cost' in info:
            info['reset_cost'] = self.reset_cost
        else:
            info['reset_cost'] = self.reset_cost
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

# Use the wrapper
env = ResetCostWrapper(gym.make('HalfCheetah-v5', render_mode=None))

#env = gym.make('HalfCheetah-v5', render_mode=None)  # Change to "human" for visualization

state_size = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]  # Continuous action space

Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states'])

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, x):
        std = np.sqrt(self.var + 1e-8)
        return (x - self.mean) / std

def train(epochs=100, num_rollouts=50):
    mean_total_rewards = []
    global_rollout = 0

    state_rms = RunningMeanStd(shape=(state_size,))  # Initialize state normalization

    initial_lr = 3e-4  # Initial learning rate
    critic_optimizer = Adam(critic.parameters(), lr=initial_lr, weight_decay=3e-3)

    for epoch in range(epochs):
        rollouts = []
        rollout_total_rewards = []

        # Calculate annealed learning rate
        lr = initial_lr * (1 - epoch / epochs)
        for param_group in critic_optimizer.param_groups:
            param_group['lr'] = lr

        for _ in range(num_rollouts):
            state, _ = env.reset()
            done = False
            samples = []

            while not done:
                state_rms.update(np.array([state]))  # Update running mean and std
                norm_state = state_rms.normalize(state)  # Normalize state
                action = get_action(norm_state)  # Use normalized state

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                samples.append((state, action, reward, next_state))
                state = next_state

            states, actions, rewards, next_states = zip(*samples)
            states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
            next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_states])
            actions = torch.stack([torch.tensor(a, dtype=torch.float32) for a in actions])
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states))
            rollout_total_rewards.append(rewards.sum().item())
            global_rollout += 1

        update_agent(rollouts, state_rms)  # Pass state_rms to update_agent
        mean_reward = np.mean(rollout_total_rewards)
        print(f'Epoch {epoch}, Mean Reward: {mean_reward}, LR: {lr}')
        mean_total_rewards.append(mean_reward)

    plt.plot(mean_total_rewards)
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

def update_critic(advantages, states):  # Add states as an argument
    values = critic(states).squeeze()
    target_values = values + advantages
    loss = 0.5 * (target_values - values) ** 2
    loss = 0.5 * (advantages ** 2).mean()
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()

max_d_kl = 0.01

def update_agent(rollouts, state_rms):  # Add state_rms as an argument
    states = torch.cat([r.states for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0)

    # Normalize states
    norm_states = torch.tensor(state_rms.normalize(states.numpy()), dtype=torch.float32)
    norm_next_states = torch.tensor(state_rms.normalize(torch.cat([r.next_states[-1].unsqueeze(0) for r in rollouts], dim=0).numpy()), dtype=torch.float32)

    advantages = []
    for i, r in enumerate(rollouts):
        norm_r_states = torch.tensor(state_rms.normalize(r.states.numpy()), dtype=torch.float32)
        advantages.append(estimate_advantages(norm_r_states, norm_next_states[i], r.rewards))  # Use normalized states
    advantages = torch.cat(advantages, dim=0).flatten()
    print("adv shape", advantages.shape)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    update_critic(advantages, norm_states)  # Pass normalized states to update_critic

    dist = Normal(actor(norm_states), torch.exp(log_std).expand_as(actor(norm_states)))  # Use normalized states
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
        dist_new = Normal(actor(norm_states), torch.exp(log_std).expand_as(actor(norm_states)))  # Use normalized states
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
'''
# using TD(0) (1stp bootstrapping)
def estimate_advantages(states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
    next_values = torch.zeros_like(rewards)
    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + 0.99 * last_value
    return next_values - values
'''
#using GAE
def estimate_advantages(states, last_state, rewards, gamma=0.99, lambda_=0.95):
    values = critic(states).squeeze()
    last_value = critic(last_state.unsqueeze(0)).squeeze()
    values = torch.cat([values, last_value.unsqueeze(0)], dim=0)

    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    advantage = 0
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantage = delta + gamma * lambda_ * advantage
        advantages[t] = advantage
    return advantages.unsqueeze(1)


def surrogate_loss(new_probs, old_probs, advantages):
    return (new_probs / old_probs * advantages).mean()

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

train(epochs=200, num_rollouts=10)
env.close()

env=gym.make('Humanoid-v5', render_mode="human")
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