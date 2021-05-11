import argparse

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess, init_weights, plot_result
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='CartPole-v0', choices=['CartPole-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
	'CartPole-v0': config.CartPole
}

if __name__ == '__main__':
	args = parser.parse_args()

	# Initialize environment and config.
	env = gym.make(args.env)
	env_config = ENV_CONFIGS[args.env]

	# Initialize deep Q-networks.
	dqn = DQN(env_config=env_config).to(device)
	dqn.apply(init_weights)

	# Create and initialize target Q-network.
	target_dqn = DQN(env_config=env_config).to(device)
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
		episode_return = 0
		episode_steps = 0

		obs = preprocess(env.reset(), env=args.env).unsqueeze(0) # Tensor [1,4]
		
		while not done:
			total_steps += 1
			episode_steps += 1

			# Get action from DQN.
			action = dqn.act(obs, eps).item()

			# Act in the true environment.
			next_obs, reward, done, info = env.step(action)
			episode_return += reward

			# Preprocess incoming observation.
			if not done:
				# Preprocess the non-terminal state.
				next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
			else:
				# Set to None the terminal state.
				next_obs = None
			
			# Add the transition to the replay memory.
			memory.push(obs, torch.as_tensor(action), next_obs, torch.as_tensor(reward))
			
			# Run DQN.optimize() every env_config["train_frequency"] steps.
			if total_steps % env_config["train_frequency"] == 0:
				loss = optimize(dqn, target_dqn, memory, optimizer)
				episode_loss += loss

			# Update the target network every env_config["target_update_frequency"] steps.
			if total_steps % env_config["target_update_frequency"] == 0:
				target_dqn.load_state_dict(dqn.state_dict())

			# Update the current observation.
			obs = next_obs
		
			# Update epsilon after each step.
			epsilon_list.append(eps)
			if eps > eps_end:
				eps -= eps_step

		# Evaluate the current agent.
		if episode % args.evaluate_freq == 0:
			mean_return = evaluate_policy(dqn, env, env_config, args, eps, n_episodes=args.evaluation_episodes)
			
			print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

			# Save current agent if it has the best performance so far.
			if mean_return >= best_mean_return:
				best_mean_return = mean_return

				print('Best performance so far! Saving model.')
				torch.save(dqn, f'models/{args.env}_best.pt')
		

		loss_list.append(episode_loss/episode_steps)
		return_list.append(episode_return)
		step_list.append(episode_steps)
		
	# Close environment after training is completed.
	env.close()


	# Plot
	x_axis = range(env_config['n_episodes'])
	plot_result(x_axis, return_list, "Episodes", "Return", "Return vs. Number of episodes")
	plot_result(x_axis, loss_list, "Episodes", "Loss", "Loss vs. Number of episodes")
	plot_result(x_axis, step_list, "Episodes", "Number of steps", "Number of steps per episode")
	plot_result(range(total_steps), epsilon_list, "Steps", "Epsilon", "Epsilon values at each step of the training")
