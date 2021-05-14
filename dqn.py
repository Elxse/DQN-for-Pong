import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def __len__(self):
		return len(self.memory)

	def push(self, obs, action, next_obs, reward):
		if len(self.memory) < self.capacity:
			self.memory.append(None)

		self.memory[self.position] = (obs, action, next_obs, reward)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		"""
		Samples batch_size transitions from the replay memory and returns a tuple
			(obs, action, next_obs, reward)
		"""
		sample = random.sample(self.memory, batch_size)
		return tuple(zip(*sample)) # ((obs1, obs2), (action1, action2), ...)


class DQN(nn.Module):
	def __init__(self, env_name, env_config):
		super(DQN, self).__init__()
		
		self.env_name = env_name
		
		# Save hyperparameters needed in the DQN class.
		self.batch_size = env_config["batch_size"]
		self.gamma = env_config["gamma"]
		self.eps_start = env_config["eps_start"]
		self.eps_end = env_config["eps_end"]
		self.anneal_length = env_config["anneal_length"]
		self.n_actions = env_config["n_actions"]
		
		if self.env_name == 'CartPole-v0':
			self.fc1 = nn.Linear(4, 256)
			self.fc2 = nn.Linear(256, self.n_actions)
		elif self.env_name == 'Pong-v0':
			self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
			self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
			self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
			self.fc1 = nn.Linear(3136, 512)
			self.fc2 = nn.Linear(512, self.n_actions)
			
		self.relu = nn.ReLU()
		self.flatten = nn.Flatten()

	def forward(self, x):
		"""Runs the forward pass of the NN depending on architecture."""
		
		if self.env_name == 'CartPole-v0':
			x = self.relu(self.fc1(x))
			x = self.fc2(x)
		if self.env_name == 'Pong-v0':
			x = self.relu(self.conv1(x))
			x = self.relu(self.conv2(x))
			x = self.relu(self.conv3(x))
			x = self.flatten(x)
			x = self.relu(self.fc1(x))
			x = self.fc2(x)
		
		return x

	def act(self, observation, epsilon, exploit=False):
		"""Selects an action with an epsilon-greedy exploration strategy.
		
		Args:
			observation (tensor): observation tensor of shape [batch_size, state_dimension] (e.g. [32, 4])
			epsilon (int)
			exploit (bool): if True, acts greedily

		Returns:
			actions (tensor): tensor of actions according to the epsilon-greedy strategy, of shape [batch_size, 1]
		"""
		
		batch_size = observation.shape[0]
		rand_value = random.random()

		if exploit or rand_value > epsilon:
			# Choose the action which gives the largest Q-values for each observation
			with torch.no_grad():
				output = self.forward(observation)
				#output = output[:,2:]
				actions = torch.argmax(output, dim=1)
				#if actions.item() == 0 or actions.item() == 2:
				#	actions[:] = 2 # right
				#else:
				#	actions[:] = 3 # left
		else:
			# Choose a random action for each observation
			random_actions = [random.randrange(self.n_actions) for _ in range(batch_size)]
			#random_actions = [random.randrange(2,4) for _ in range(batch_size)]
			actions = torch.tensor(random_actions)

		return actions.unsqueeze(1)

def optimize(dqn, target_dqn, memory, optimizer):
	"""This function samples a batch from the replay buffer and optimizes the Q-network."""

	# If we don't have enough transitions stored yet, we don't train.
	if len(memory) < dqn.batch_size:
		return 0

	# TODO: Sample a batch from the replay memory and concatenate so that there are
	#	   four tensors in total: observations, actions, next observations and rewards.
	#	   Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
	#	   Note that special care is needed for terminal transitions!
	observations, actions, next_observations, rewards = memory.sample(dqn.batch_size)

	# Transform every tuple into tensors and move it to GPU if it is available	
	observations = torch.cat(observations)
	observations = observations.to(device)

	non_terminal_next_observations = [next_obs for next_obs in next_observations if next_obs is not None]
	non_terminal_next_observations = torch.cat(non_terminal_next_observations).float()
	non_terminal_next_observations = non_terminal_next_observations.to(device)
	
	actions = torch.stack(actions, dim=0)
	actions = torch.unsqueeze(actions, 1)
	actions = actions.to(device)

	rewards = torch.stack(rewards, dim=0)
	rewards = torch.unsqueeze(rewards, 1)
	rewards = rewards.to(device)

	# TODO: Compute the current estimates of the Q-values for each state-action
	#	   pair (s,a). Here, torch.gather() is useful for selecting the Q-values
	#	   corresponding to the chosen actions.
	output = dqn.forward(observations)
	output = output.to(device)
	q_values = torch.gather(input=output, index=actions, dim=1)

	# TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
	non_terminal_mask = torch.BoolTensor(list(map(lambda obs: obs is not None, next_observations))) # indices for non terminal transitions
	terminal_mask = ~non_terminal_mask # indices for terminal transitions
	target_dqn_qpred = target_dqn.forward(non_terminal_next_observations) 
	target_dqn_qpred_max = torch.max(target_dqn_qpred, axis=1)[0].unsqueeze(1)
	
	q_value_targets = torch.zeros(dqn.batch_size, 1)
	q_value_targets = q_value_targets.to(device)
	q_value_targets[non_terminal_mask] = rewards[non_terminal_mask] + dqn.gamma * target_dqn_qpred_max
	q_value_targets[terminal_mask] = rewards[terminal_mask]
	
	# Compute loss.
	loss = F.mse_loss(q_values, q_value_targets)

	# Perform gradient descent.
	optimizer.zero_grad()

	loss.backward()
	optimizer.step()
	
	return loss.item()
