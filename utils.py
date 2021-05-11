import random

import gym
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v0']:
        return torch.tensor(obs, device=device).float()
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')

def init_weights(model):
	if isinstance(model, torch.nn.Linear):
		torch.nn.init.normal_(model.weight, mean=0, std=0.01)
		torch.nn.init.zeros_(model.bias)

def plot_result(x, y, xlabel, ylabel, title):
	plt.plot(x, y)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()