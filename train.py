import argparse

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess
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


def init_weights(model):
	if isinstance(model, torch.nn.Linear):
		torch.nn.init.normal_(model.weight, mean=0, std=0.01)
		torch.nn.init.zeros_(model.bias)

def plot_result(x, y, xlabel, ylabel):
	plt.plot(x, y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()


if __name__ == '__main__':
	args = parser.parse_args()

	# Initialize environment and config.
	env = gym.make(args.env)
	env_config = ENV_CONFIGS[args.env]

	# Initialize deep Q-networks.
	dqn = DQN(env_config=env_config).to(device)
	dqn.apply(init_weights)

	# TODO: Create and initialize target Q-network.
	target_dqn = DQN(env_config=env_config).to(device)
	target_dqn.load_state_dict(dqn.state_dict())

	# Create replay memory.
	memory = ReplayMemory(env_config['memory_size'])
	
	# Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
	optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

	# Keep track of best evaluation mean return achieved so far.
	best_mean_return = -float("Inf")

	eps = dqn.eps_start
	eps_end = dqn.eps_end
	eps_step = dqn.eps_step

	losses_list, reward_list, step_list, epsilon_list = [], [], [], []

	for episode in range(env_config['n_episodes']):
		done = False
		
		losses = 0
		total_reward = 0
		t = 0 # Number of steps
		
		obs = preprocess(env.reset(), env=args.env).unsqueeze(0) # Tensor of shape (1,4)
		
		while not done:
			t += 1

			# TODO: Get action from DQN.
			action = dqn.act(obs, eps).item()

			# Act in the true environment.
			next_obs, reward, done, info = env.step(action)
			total_reward += reward

			# Preprocess incoming observation.
			if not done:
				next_obs = preprocess(next_obs, env=args.env).unsqueeze(0) # tensor
				#if next_obs.shape[0] == 4:
				#	next_obs = torch.unsqueeze(next_obs, 0)
				#	next_obs = next_obs.to(device)
			else:
				next_obs = None
			
			# TODO: Add the transition to the replay memory. Remember to convert
			#       everything to PyTorch tensors!
			memory.push(obs, torch.as_tensor(action), next_obs, torch.as_tensor(reward))
			
			# TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
			if t % env_config["train_frequency"] == 0:
				#print("hre")
				loss = optimize(dqn, target_dqn, memory, optimizer)
				losses += loss

			# TODO: Update the target network every env_config["target_update_frequency"] steps.
			if t % env_config["target_update_frequency"] == 0:
				target_dqn.load_state_dict(dqn.state_dict())

			obs = next_obs

		# Evaluate the current agent.
		if episode % args.evaluate_freq == 0:
			mean_return = evaluate_policy(dqn, env, env_config, args, eps, n_episodes=args.evaluation_episodes)
			
			print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

			# Save current agent if it has the best performance so far.
			if mean_return >= best_mean_return:
				best_mean_return = mean_return

				print('Best performance so far! Saving model.')
				torch.save(dqn, f'models/{args.env}_best.pt')
		
		# Update after each episode
		if eps > eps_end:
			eps -= eps_step

		losses_list.append(losses/t)
		reward_list.append(total_reward)
		#step_list.append(t)
		epsilon_list.append(eps)

		
		#print(f"t = {t}")
		
	# Close environment after training is completed.
	env.close()


	# Plot
	x_axis = range(env_config['n_episodes'])
	plot_result(x_axis, reward_list, "Episodes", "Reward")
	#plot_result(x_axis, losses_list, "Episodes", "Losses")
	#plot_result(x_axis, step_list, "Episodes", "Number of steps")
	#plot_result(x_axis, epsilon_list, "Episodes", "Epsilon")
