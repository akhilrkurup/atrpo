import argparse
import gymnasium as gym
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
#from models.mlp_policy_disc import DiscretePolicy
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent import Agent


parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="HalfCheetah-v5", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.5, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=3*1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=5000, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=5000, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=40, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=10, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""

env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.reset(seed=args.seed) 

"""define actor and critic"""
if args.model_path is None:
    # if is_disc_action:
    #     policy_net = DiscretePolicy(state_dim, env.action_space.n)
    # else:
    policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
    value_net = Value(state_dim)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
policy_net.to(device)
value_net.to(device)

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=args.num_threads)


def update_params(batch):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform TRPO update"""
    trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)

def evaluate_policy(env, agent, max_steps, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = agent.running_state(state)
        total_reward = 0
        for _ in range(max_steps):
            state_var = torch.from_numpy(state).unsqueeze(0).to(dtype).to(device)
            with torch.no_grad():
               action_mean, _, _ = agent.policy(state_var)
               action = action_mean.cpu().numpy()[0]

               

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = agent.running_state(next_state)
            total_reward += reward
            state = next_state
            # if done or truncated:
            #     print(next_state)
            #     break
        total_rewards.append(total_reward)
    return np.mean(total_rewards)

def main_loop():
    
    eval_steps = []
    returns_1k = []
    returns_10k = []
    total_env_steps=0
    for i_iter in range(args.max_iter_num):
        if(total_env_steps>=1e7):break
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)

        total_env_steps += log['num_steps']
        #print(total_env_steps)
        t0 = time.time()
        update_params(batch)
        t1 = time.time()
        """evaluate with determinstic action (remove noise for exploration)"""
        _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
        t2 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, t2-t1, log['min_reward'], log['max_reward'], log['avg_reward'], log_eval['avg_reward']))
        
        if total_env_steps // 100000 > (total_env_steps - log['num_steps']) // 100000:
            avg_r_1k = evaluate_policy(env, agent, max_steps=1000)
            avg_r_10k = evaluate_policy(env, agent, max_steps=10000)
            eval_steps.append(total_env_steps)
            returns_1k.append(avg_r_1k)
            returns_10k.append(avg_r_10k)
            print(f'[Eval @ {int(total_env_steps)} steps] Return (max 1k): {avg_r_1k:.2f}, Return (max 10k): {avg_r_10k:.2f}')


        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_trpo.p'.format(args.env_name)), 'wb'))
            to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()

       # Plot after training
    plt.figure(figsize=(10, 6))
    plt.plot(eval_steps, returns_1k, label='Max 1,000 steps', marker='o')
    plt.plot(eval_steps, returns_10k, label='Max 10,000 steps', marker='x')
    plt.xlabel('Environment Steps')
    plt.ylabel('Average Undiscounted Return')
    plt.title(f'Evaluation Returns - {args.env_name}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(assets_dir(), f'plots/{args.env_name}_eval_plot.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.show()    


main_loop()