import argparse

import gym
from gym.wrappers import AtariPreprocessing
import torch
import torch.nn as nn

import config
from utils import preprocess, init_weights, plot_result
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='Pong-v0', choices=['CartPole-v0', 'Pong-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--model_name', default='Pong-v0', help='Model to load if present.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'Pong-v0': config.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    if args.env == "Pong-v0":
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30, scale_obs=True)
        # Now every obs is numpy array of shape (84,84)
    env_config = ENV_CONFIGS[args.env]

    # Load Model
    try:
        model_name = str(args.model_name) + '_best.pt'
        target_model_name = str(args.model_name) + '_target.pt'
        dqn = torch.load(f'models/{model_name}')
        target_dqn = torch.load(f'models/{target_model_name}')
    except:
        # Initialize deep Q-networks.
        dqn = DQN(env_name=args.env, env_config=env_config).to(device)
        dqn.apply(init_weights)

        # Create and initialize target Q-network.
        target_dqn = DQN(env_name=args.env, env_config=env_config).to(device)
        target_dqn.load_state_dict(dqn.state_dict())

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    # Compute the value for epsilon linear annealing.
    eps = dqn.eps_start
    eps_end = dqn.eps_end
    anneal_length = dqn.anneal_length
    eps_step = (eps - eps_end) / anneal_length

    # Initialize lists to keep track of episodes' loss, return, number of steps, and epsilon values throughout the training
    loss_list, return_list, step_list, epsilon_list = [], [], [], []

    total_steps = 0

    for episode in range(env_config['n_episodes']):
        done = False

        episode_loss = 0
        episode_rewards = []
        episode_steps = 0


        obs = preprocess(env.reset(), env=args.env).unsqueeze(0) # [1, 84, 84]
        obs_stack = torch.cat(env_config["obs_stack_size"] * [obs]).unsqueeze(0).to(device)  # [1, 4, 84, 84]

        while not done:
            total_steps += 1
            episode_steps += 1

            # Get action from DQN.
            if args.env == "Pong-v0":
                action = dqn.act(obs_stack, eps).item()
            else:
                action = dqn.act(obs, eps).item()

            # Act in the true environment.
            next_obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

            # Preprocess incoming observation.
            if not done:
                # Preprocess the non-terminal state.
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
                if args.env == "Pong-v0":
                    next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
                    next_obs_stack = preprocess(next_obs_stack, env=args.env)
            else:
                # Set to None the terminal state.
                next_obs = None
                if args.env == "Pong-v0":
                    next_obs_stack = None

            # Add the transition to the replay memory.
            if args.env == "CartPole-v0":
                memory.push(obs, torch.as_tensor(action), next_obs, torch.as_tensor(reward))
            if args.env == "Pong-v0":
                memory.push(obs_stack, torch.as_tensor(action), next_obs_stack, torch.as_tensor(reward))

            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if total_steps % env_config["train_frequency"] == 0:
                loss = optimize(dqn, target_dqn, memory, optimizer)
                episode_loss += loss

            # Update the target network every env_config["target_update_frequency"] steps.
            if total_steps % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            # Update the current observation.
            obs = next_obs
            obs_stack = next_obs_stack

            # Update epsilon after each step.
            epsilon_list.append(eps)
            if eps > eps_end:
                eps -= eps_step

        # Compute the episode return
        episode_return = sum([env_config['gamma'] ** t * episode_rewards[t] for t in range(len(episode_rewards))])

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, eps, n_episodes=args.evaluation_episodes)

            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
                torch.save(target_dqn, f'models/{args.env}_target.pt')

        loss_list.append(episode_loss/episode_steps)
        return_list.append(episode_return)
        step_list.append(episode_steps)

    # Close environment after training is completed.
    env.close()


    # Plot
    x_axis = range(env_config['n_episodes'])
    plot_result(x_axis, return_list, "Episodes", "Return", title="Return vs. Number of episodes", save_path=f"results/{args.env}", env_name=args.env)
    plot_result(x_axis, loss_list, "Episodes", "Loss", title="Loss vs. Number of episodes", save_path=f"results/{args.env}", env_name=args.env)
    plot_result(x_axis, step_list, "Episodes", "Number of steps", title="Number of steps per episode", save_path=f"results/{args.env}", env_name=args.env)
    plot_result(range(total_steps), epsilon_list, "Steps", "Epsilon", title="Epsilon values at each step of the training", save_path=f"results/{args.env}", env_name=args.env)
