import random
from collections import deque
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.obs_dim = obs_dim
        self.pointer = 0
        self.size = 0

        self.obs = torch.zeros((capacity, obs_dim),dtype=torch.float,device=self.device)
        self.actions = torch.zeros((capacity, 1),dtype=torch.long,device=self.device)
        self.rewards = torch.zeros((capacity, 1),dtype=torch.float,device=self.device)
        self.next_obs = torch.zeros((capacity, obs_dim),dtype=torch.float,device=self.device)
        self.dones = torch.zeros((capacity, 1),dtype=torch.bool,device=self.device)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pointer] = torch.from_numpy(obs).to(self.device)
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_obs[self.pointer] = torch.from_numpy(next_obs).to(self.device)
        self.dones[self.pointer] = done

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        return self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx], self.dones[idx]

    def __len__(self):
        return self.size