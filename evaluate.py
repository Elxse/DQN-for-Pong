import argparse
import random

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='Pong-v0', choices=['CartPole-v0', 'Pong-v0'])
parser.add_argument('--path', type=str, help='Path to stored DQN model.')
parser.add_argument('--n_eval_episodes', type=int, default=1, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--render', dest='render', action='store_true', help='Render the environment.')
parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save the episodes as video.')
parser.set_defaults(render=False)
parser.set_defaults(save_video=False)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
	'CartPole-v0': config.CartPole,
	'Pong-v0': config.Pong
}

def evaluate_policy(dqn, env, env_config, args, eps_start, n_episodes, discounted=False, render=False, verbose=False):
	"""Runs {n_episodes} episodes to evaluate current policy."""
	total_return = 0

	for i in range(n_episodes):
		obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
		if args.env == "Pong-v0":
			obs_stack = torch.cat(env_config["obs_stack_size"] * [obs]).unsqueeze(0).to(device)

		done = False
		rewards_list = []

		while not done:
			if render:
				env.render()
				
			if args.env == "Pong-v0":
				action = dqn.act(obs_stack, eps_start, exploit=True).item()
			else:
				action = dqn.act(obs, eps_start, exploit=True).item()
			
			obs, reward, done, info = env.step(action)
			obs = preprocess(obs, env=args.env).unsqueeze(0)
			if args.env == "Pong-v0":
				obs_stack = torch.cat((obs_stack[:, 1:, ...], obs.unsqueeze(1)), dim=1).to(device)
				obs_stack = preprocess(obs_stack, env=args.env)
			
			rewards_list.append(reward)

		if discounted:
			episode_return = sum([env_config['gamma']**t * rewards_list[t] for t in range(len(rewards_list))])
		else:
			episode_return = sum(rewards_list)
		total_return += episode_return
		
		if verbose:
			print(f'Finished episode {i+1} with a total return of {episode_return}')

	
	return total_return / n_episodes

if __name__ == '__main__':
	args = parser.parse_args()

	# Initialize environment and config
	env = gym.make(args.env)
	env_config = ENV_CONFIGS[args.env]

	if args.save_video:
		env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: True, force=True)

	# Load model from provided path.
	dqn = torch.load(args.path, map_location=torch.device('cpu'))
	dqn.eval()

	mean_return = evaluate_policy(dqn, env, env_config, args, dqn.eps_start, args.n_eval_episodes, render=args.render and not args.save_video, verbose=True)
	print(f'The policy got a mean return of {mean_return} over {args.n_eval_episodes} episodes.')

	env.close()