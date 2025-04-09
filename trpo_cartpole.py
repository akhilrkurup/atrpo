import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple

# Updated environment name
env = gym.make('CartPole-v1',render_mode=None)

state_size = env.observation_space.shape[0]
num_actions = env.action_space.n
print(state_size, num_actions)

Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states'])

def train(epochs=100, num_rollouts=20, render_frequency=None):
    mean_total_rewards = []
    global_rollout = 0

    for epoch in range(epochs):
        rollouts = []
        rollout_total_rewards = []

        for _ in range(num_rollouts):
            state, _ = env.reset()  # Updated to handle new Gym reset format
            done = False
            samples = []

            while not done:
                with torch.no_grad():
                    action = get_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated  # Updated done condition

                # Collect samples
                samples.append((state, action, reward, next_state))
                state = next_state

            # Transpose samples
            states, actions, rewards, next_states = zip(*samples)

            states = torch.tensor(np.array(states), dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            actions = torch.tensor(actions).unsqueeze(1)
            rewards = torch.tensor(rewards).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states))
            rollout_total_rewards.append(rewards.sum().item())
            global_rollout += 1

        update_agent(rollouts)
        mean_reward = np.mean(rollout_total_rewards)
        print(f'Epoch {epoch}: Mean total reward: {mean_reward}')
        mean_total_rewards.append(mean_reward)

    plt.plot(mean_total_rewards)
    plt.show()

# Define Actor Network
actor_hidden = 32
actor = nn.Sequential(
    nn.Linear(state_size, actor_hidden),
    nn.ReLU(),
    nn.Linear(actor_hidden, num_actions),
    nn.Softmax(dim=1),
)

def get_action(state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    dist = Categorical(actor(state))
    return dist.sample().item()

# Define Critic Network
critic_hidden = 32
critic = nn.Sequential(
    nn.Linear(state_size, critic_hidden),
    nn.ReLU(),
    nn.Linear(critic_hidden, 1),
)
critic_optimizer = Adam(critic.parameters(), lr=0.005)

def update_critic(advantages):
    loss = 0.5 * (advantages ** 2).mean()  # MSE Loss
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()

# Trust Region Policy Optimization (TRPO) updates
max_d_kl = 0.01

def update_agent(rollouts):
    states = torch.cat([r.states for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()
    #print("states shape",states.shape,"action shape", actions.shape)

    advantages = [
        estimate_advantages(states, next_states[-1], rewards)
        for states, _, rewards, next_states in rollouts
    ]
    advantages = torch.cat(advantages, dim=0).flatten()
    #print("adv shape",advantages.shape)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages

    update_critic(advantages)

    distribution = actor(states)
    #print("disttribution shape",distribution.shape)
    probabilities = distribution[range(distribution.shape[0]), actions]


    # Compute TRPO updates
    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(distribution, distribution)

    parameters = list(actor.parameters())

    g = flat_grad(L, parameters, retain_graph=True)
    d_kl = flat_grad(KL, parameters, create_graph=True)

    def HVP(v):
        return flat_grad(d_kl @ v, parameters, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g)
    max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir

    def criterion(step):
        apply_update(step)

        with torch.no_grad():
            new_distribution = actor(states)
            probabilities_new = new_distribution[range(new_distribution.shape[0]), actions]

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(distribution, new_distribution)

        L_improvement = L_new - L

        if L_improvement > 0 and KL_new <= max_d_kl:
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

def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (new_probabilities / old_probabilities * advantages).mean()

def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()

def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    grads = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    return torch.cat([g.view(-1) for g in grads])

def conjugate_gradient(A, b, delta=0., max_iterations=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    for _ in range(max_iterations):
        AVP = A(p)
        alpha = (r @ r) / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        r_new = r - alpha * AVP
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p

        x = x_new
        r = r_new

    return x

def apply_update(grad_flattened):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel

# Train agent
train(epochs=50, num_rollouts=20)
env.close()

env=gym.make('CartPole-v1', render_mode="human")
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