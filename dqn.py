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
	def __init__(self, env_config):
		super(DQN, self).__init__()

		# Save hyperparameters needed in the DQN class.
		self.batch_size = env_config["batch_size"]
		self.gamma = env_config["gamma"]
		self.eps_start = env_config["eps_start"]
		self.eps_end = env_config["eps_end"]
		self.anneal_length = env_config["anneal_length"]
		self.eps_step = (self.eps_start - self.eps_end) / self.anneal_length 
		self.n_actions = env_config["n_actions"]

		self.fc1 = nn.Linear(4, 256)
		self.fc2 = nn.Linear(256, self.n_actions)

		self.relu = nn.ReLU()
		self.flatten = nn.Flatten()

	def forward(self, x):
		"""Runs the forward pass of the NN depending on architecture."""
		x = self.relu(self.fc1(x))
		x = self.fc2(x)

		return x

	def act(self, observation, epsilon, exploit=False):
		"""Selects an action with an epsilon-greedy exploration strategy."""
		# TODO: Implement action selection using the Deep Q-network. This function
		#       takes an observation tensor and should return a tensor of actions.
		#       For example, if the state dimension is 4 and the batch size is 32,
		#       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
		# TODO: Implement epsilon-greedy exploration.
		
		batch_size = observation.shape[0]
		rand_value = random.random()
		if exploit or rand_value > epsilon:
			with torch.no_grad():
				output = self.forward(observation)
			actions = torch.argmax(output, dim=1) 
			#print(f"output = {output}")
			#print(f"actions = {actions}")
		else:
			random_actions = [random.randrange(self.n_actions) for _ in range(batch_size)]
			actions = torch.tensor(random_actions)
			#print(f"actions = {actions}")
		return actions.unsqueeze(1)

def optimize(dqn, target_dqn, memory, optimizer):
	"""This function samples a batch from the replay buffer and optimizes the Q-network."""
	# If we don't have enough transitions stored yet, we don't train.
	if len(memory) < dqn.batch_size:
		return 0

	# TODO: Sample a batch from the replay memory and concatenate so that there are
	#       four tensors in total: observations, actions, next observations and rewards.
	#       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
	#       Note that special care is needed for terminal transitions!
	observations, actions, next_observations, rewards = memory.sample(dqn.batch_size)
	
	observations = torch.cat(observations)  # [32, 4]
	observations = observations.to(device)

	non_terminal_next_observations = [next_obs for next_obs in next_observations if next_obs is not None]
	#print(len(non_terminal_next_observations))
	non_terminal_next_observations = torch.cat(non_terminal_next_observations).float()
	non_terminal_next_observations = non_terminal_next_observations.to(device)

	actions = torch.stack(actions, dim=0)
	actions = torch.unsqueeze(actions, 1) # [32, 1]
	actions = actions.to(device)

	rewards = torch.stack(rewards, dim=0)
	rewards = torch.unsqueeze(rewards, 1)
	rewards = rewards.to(device)

	# TODO: Compute the current estimates of the Q-values for each state-action
	#       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
	#       corresponding to the chosen actions.
	output = dqn.forward(observations) # [32, 2]
	output = output.to(device)
	q_values = torch.gather(input=output, index=actions, dim=1)
	#print(f"output = {output}\n, actions = {actions}\n, q_values = {q_values}\n")

	# TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
	non_terminal_mask = torch.BoolTensor(list(map(lambda obs: obs is not None, next_observations))) # indices for non terminal transitions
	terminal_mask = ~non_terminal_mask # indices for terminal transitions
	target_dqn_qpred = target_dqn.forward(non_terminal_next_observations) 
	target_dqn_qpred_max = torch.max(target_dqn_qpred, axis=1)[0].unsqueeze(1)
	#print(target_dqn_qpred)
	#print(target_dqn_qpred_max)
	#print(rewards[non_terminal_mask])
	q_value_targets = torch.zeros(dqn.batch_size, 1)
	q_value_targets[non_terminal_mask] = rewards[non_terminal_mask] + dqn.gamma * target_dqn_qpred_max
	q_value_targets[terminal_mask] = rewards[terminal_mask]
	
	#if not isTerminal:
	#	q_value_targets = rewards + dqn.gamma * torch.max(target_dqn.forward(next_observations))
	#else:
	#	q_value_targets = rewards
	
	# Compute loss.
	#loss = F.mse_loss(q_values.squeeze(), q_value_targets)
	loss = F.mse_loss(q_values, q_value_targets)
	#print(f"loss = {loss}")

	# Perform gradient descent.
	optimizer.zero_grad()

	loss.backward()
	optimizer.step()
	
	return loss.item()
