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
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='CartPole-v0', choices=['CartPole-v0', 'Pong-v0'])
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
    env_config = ENV_CONFIGS[args.env]
    params = f"lr_{env_config['lr']}_gamma_{env_config['gamma']}_tg_{env_config['target_update_frequency']}_ann_{env_config['anneal_length']}"

    # Load Model.
    try:
        model_name = f"{args.env}_best_{params}_ax.pt"
        target_model_name = f"{args.env}_target_{params}_ax.pt"
        dqn = torch.load(f'models/{model_name}')
        target_dqn = torch.load(f'models/{target_model_name}')
        print("Loading model...")
    except:
        print("Initialize model...")

        # Initialize deep Q-networks.
        dqn = DQN(env_name=args.env, env_config=env_config).to(device)
        dqn.apply(init_weights)

        # Create and initialize target Q-network.
        target_dqn = DQN(env_name=args.env, env_config=env_config).to(device)
        target_dqn.load_state_dict(dqn.state_dict())
    
    # Load results pickle file to add new results from the pretrained model.
    results_file = f"results/{args.env}/{args.env}_results_{params}.pkl"
    try:
        with open(results_file, 'rb') as file:
            results_dict = pickle.load(file)
        print("Load pickle file")
        return_list = results_dict['return']
        eval_return_list = results_dict['eval_return']
        loss_list = results_dict['loss']
        step_list = results_dict['step']
        epsilon_list = results_dict['epsilon']
        dqn.eps_start = results_dict['eps_start']
        best_mean_return = results_dict['best_mean_return']
        memory = results_dict['memory']
        print(f"best_mean_return = {best_mean_return}")
    except:
        # Initialize lists to keep track of episodes' loss, return, number of steps, and epsilon values throughout the training.
        loss_list, return_list, eval_return_list, step_list, epsilon_list = [], [], [], [], []

        # Keep track of best evaluation mean return achieved so far.
        best_mean_return = -float("Inf")

        # Create replay memory.
        memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Compute the value for epsilon linear annealing.
    eps = dqn.eps_start
    eps_end = dqn.eps_end
    anneal_length = dqn.anneal_length
    eps_step = (eps - eps_end) / anneal_length
    
	# Keep track of the total number of steps.
    total_steps = 0

    for episode in range(env_config['n_episodes']):
        done = False

        episode_loss = 0
        episode_rewards = []
        episode_steps = 0

        obs = preprocess(env.reset(), env=args.env).unsqueeze(0).to(device)
        if args.env == "Pong-v0":
            obs_stack = torch.cat(env_config["obs_stack_size"] * [obs]).unsqueeze(0).to(device)

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
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0).to(device)
                if args.env == "Pong-v0":
                    next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
                    next_obs_stack = preprocess(next_obs_stack, env=args.env).to(device)
            else:
                # Set to None the terminal state.
                next_obs = None
                if args.env == "Pong-v0":
                    next_obs_stack = None

            # Add the transition to the replay memory. Everything has been move to GPU if available.
            action = torch.as_tensor(action).to(device)
            reward = torch.as_tensor(reward).to(device)
            if args.env == "CartPole-v0":
                memory.push(obs, action, next_obs, reward)
            if args.env == "Pong-v0":
                memory.push(obs_stack, action, next_obs_stack, reward)

            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if total_steps % env_config["train_frequency"] == 0:
                loss = optimize(dqn, target_dqn, memory, optimizer)
                episode_loss += loss

            # Update the target network every env_config["target_update_frequency"] steps.
            if total_steps % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            # Update the current observation.
            obs = next_obs
            if args.env == "Pong-v0":
                obs_stack = next_obs_stack

            # Update epsilon after each step.
            epsilon_list.append(eps)
            if eps > eps_end:
                eps -= eps_step

        # Compute the episode return.
        episode_undiscounted_return = sum(episode_rewards)

	   	# Add results to lists.
        loss_list.append(episode_loss/episode_steps)
        return_list.append(episode_undiscounted_return)
        step_list.append(episode_steps)
        
        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, eps, n_episodes=args.evaluation_episodes)
            eval_return_list.append(mean_return)

            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best_{params}.pt')
                torch.save(target_dqn, f'models/{args.env}_target_{params}.pt')
            
            # Save results.
            results_dict = {
                'return': return_list,
                'eval_return': eval_return_list,
                'loss': loss_list,
                'step': step_list,
                'epsilon': epsilon_list,
                'eps_start': eps,
                'best_mean_return': best_mean_return,
                'memory': memory
            }
            print("Save pickle file")
            with open(results_file, 'wb') as file:
                pickle.dump(results_dict, file)
                    
            # Plot
            # x_axis = range(len(results_dict['return']))
            # plot_result(x_axis, results_dict['return'], "Episodes", "Return", title="Return vs. Number of episodes", save_path=f"results/{args.env}", env_name=args.env, params=params)
            # plot_result(range(len(results_dict['eval_return'])), results_dict['eval_return'], "Episodes", "Return", title="Evaluation Return every 25 episodes", save_path=f"results/{args.env}", env_name=args.env, params=params)
            # plot_result(x_axis, results_dict['loss'], "Episodes", "Loss", title="Loss vs. Number of episodes", save_path=f"results/{args.env}", env_name=args.env, params=params)
            # plot_result(x_axis, results_dict['step'], "Episodes", "Number of steps", title="Number of steps per episode", save_path=f"results/{args.env}", env_name=args.env, params=params)
            # plot_result(range(len(results_dict['epsilon'])), results_dict['epsilon'], "Steps", "Epsilon", title="Epsilon values at each step of the training", save_path=f"results/{args.env}", env_name=args.env, params=params)

    # Close environment after training is completed.
    env.close()
    
    # Save results.
    results_dict = {
        'return': return_list,
        'eval_return': eval_return_list,
        'loss': loss_list,
        'step': step_list,
        'epsilon': epsilon_list,
        'eps_start': eps,
        'best_mean_return': best_mean_return,
        'memory': memory
    }
    print("Save pickle file")
    with open(results_file, 'wb') as file:
        pickle.dump(results_dict, file)
            
    # Plot.
    x_axis = range(len(results_dict['return']))
    plot_result(x_axis, results_dict['return'], "Episodes", "Return", title="Return vs. Number of episodes", save_path=f"results/{args.env}", env_name=args.env, params=params)
    plot_result(range(len(results_dict['eval_return'])), results_dict['eval_return'], "Episodes", "Return", title="Evaluation Return every 25 episodes", save_path=f"results/{args.env}", env_name=args.env, params=params)
    plot_result(x_axis, results_dict['loss'], "Episodes", "Loss", title="Loss vs. Number of episodes", save_path=f"results/{args.env}", env_name=args.env, params=params)
    plot_result(x_axis, results_dict['step'], "Episodes", "Number of steps", title="Number of steps per episode", save_path=f"results/{args.env}", env_name=args.env, params=params)
    plot_result(range(len(results_dict['epsilon'])), results_dict['epsilon'], "Steps", "Epsilon", title="Epsilon values at each step of the training", save_path=f"results/{args.env}", env_name=args.env, params=params)