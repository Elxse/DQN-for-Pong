import argparse

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='CartPole-v0', choices=['CartPole-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
	'CartPole-v0': config.CartPole
}


def init_wb(model):
	if isinstance(model, torch.nn.Linear):
		torch.nn.init.normal_(model.weight, mean=0, std=0.01)
		torch.nn.init.zeros_(model.bias)


if __name__ == '__main__':
	args = parser.parse_args()

	# Initialize environment and config.
	env = gym.make(args.env)
	env_config = ENV_CONFIGS[args.env]

	# Initialize deep Q-networks.
	dqn = DQN(env_config=env_config).to(device)
	# wb_ini = init_wb
	dqn.apply(init_wb)

	# TODO: Create and initialize target Q-network.
	target_dqn = DQN(env_config=env_config).to(device)
	target_dqn.load_state_dict(dqn.state_dict())

	# Create replay memory.
	memory = ReplayMemory(env_config['memory_size'])
	
	# Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
	optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

	# Keep track of best evaluation mean return achieved so far.
	best_mean_return = -float("Inf")

	eps_start = env_config["eps_start"]
	eps_end = env_config["eps_end"]
	anneal_length = env_config["anneal_length"]
	eps_step = (eps_start - eps_end) / anneal_length

	for episode in range(env_config['n_episodes']):
		done = False
		t = 0 # Number of steps

		obs = preprocess(env.reset(), env=args.env).unsqueeze(0) # Tensor of shape (1,4)
		# print(obs.shape)
		
		while not done:
			# TODO: Get action from DQN.
			action = dqn.act(obs, eps_start).item()
			if eps_start >= eps_end:
				eps_start -= eps_step

			# Act in the true environment.
			#obs, reward, done, info = env.step(action)
			next_obs, reward, done, info = env.step(action) # should be next_obs right?

			# Preprocess incoming observation.
			if not done:
				#obs = preprocess(obs, env=args.env).unsqueeze(0)
				next_obs = preprocess(next_obs, env=args.env).unsqueeze(0) # tensor
			
			# TODO: Add the transition to the replay memory. Remember to convert
			#       everything to PyTorch tensors!
			#print("	OBS: ", obs)
			#print("	ACTION:", action)
			#print("	NEXT_OBS: ", next_obs)
			#print("	REWARD:", reward)
			if not torch.is_tensor(next_obs):
				next_obs = torch.from_numpy(next_obs)

			if next_obs.shape[0] == 4:
				next_obs = torch.unsqueeze(next_obs,0)
				next_obs = next_obs.to(device)

			memory.push(obs, torch.as_tensor(action), next_obs, torch.as_tensor(reward))
			
			# TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
			if t % env_config["train_frequency"] == 0:
				optimize(dqn, target_dqn, memory, optimizer, done)

			# TODO: Update the target network every env_config["target_update_frequency"] steps.
			if t % env_config["target_update_frequency"] == 0:
				target_dqn.load_state_dict(dqn.state_dict())

			t += 1
			obs = next_obs

		# Evaluate the current agent.
		if episode % args.evaluate_freq == 0:
			mean_return = evaluate_policy(dqn, env, env_config, args, eps_start, n_episodes=args.evaluation_episodes)
			
			print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

			# Save current agent if it has the best performance so far.
			if mean_return >= best_mean_return:
				best_mean_return = mean_return

				print('Best performance so far! Saving model.')
				torch.save(dqn, f'models/{args.env}_best.pt')
		
	# Close environment after training is completed.
	env.close()
