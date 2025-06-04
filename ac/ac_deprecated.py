import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        logits = self.policy(obs)
        dist = Categorical(logits=logits)
        return dist

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, num_critics, hidden_dim):
        super(Critic, self).__init__()
        self.action_dim = action_dim
        
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_critics)
        ])

    def forward(self, obs, action):
        if action.dim() == 1:
            action = F.one_hot(action, num_classes=self.action_dim).float()
        elif action.shape[-1] != self.action_dim:
            raise ValueError("Expected action to be one-hot or indices.")

        h_action = torch.cat([obs, action], dim=-1)
        return [critic(h_action) for critic in self.critics]

class ACAgent:
    def __init__(self, obs_dim, action_dim, hidden_dim, device, 
                 lr=2e-5, num_critics=2, critic_tau=0.005, gamma=0.99, alpha=0.2):
        self.device = device
        self.gamma = gamma
        self.critic_tau = critic_tau
        self.alpha = alpha
        self.action_dim = action_dim

        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(obs_dim, action_dim, num_critics, hidden_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim, num_critics, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, obs, eval_mode=False, early_bias=True):
        obs_array = np.array(obs)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)

        dist = self.actor.forward(obs_tensor)
        probs = dist.probs.clone().squeeze(0)

        availability_mask = torch.tensor(obs_array, dtype=torch.float32, device=self.device)
        probs = probs * availability_mask

        if early_bias:
            num_actions = probs.shape[-1]
            decay = torch.exp(-torch.arange(num_actions, device=self.device, dtype=torch.float32) / 30.0) # decay factor
            probs = probs * decay
            probs = probs / probs.sum()

        if eval_mode:
            action = probs.argmax()
        else:
            biased_dist = torch.distributions.Categorical(probs=probs)
            action = biased_dist.sample()
        print(action.shape, action.item())
        return action.item()

    def update_critic(self, replay_iter):
        obs, action, reward, discount, next_obs = [x.to(self.device) for x in replay_iter]
        print('Shapes', obs.shape, action.shape, reward.shape, discount.shape, next_obs.shape)
        with torch.no_grad():
            next_dist = self.actor.forward(next_obs)
            next_action = next_dist.sample()

            target_Qs = self.critic_target(next_obs, next_action)  # List of [B, 1]
            target_Qs = torch.stack(target_Qs, dim=1).squeeze(-1)  # Shape: [B, N]
            num_critics = target_Qs.shape[1]
            rand_idxs = torch.randperm(num_critics)[:2]
            Q1 = target_Qs[:, rand_idxs[0]]
            Q2 = target_Qs[:, rand_idxs[1]]
            target_Q = reward + discount * torch.min(Q1, Q2)
            target_Q = target_Q.unsqueeze(1)

        current_Qs = self.critic.forward(obs, action)
        current_Qs = torch.stack(current_Qs, dim=1).squeeze(-1)
        target_expanded = target_Q.expand_as(current_Qs)
        assert target_expanded.shape == current_Qs.shape, f"Shape mismatch: target {target_expanded.shape}, current {current_Qs.shape}"
        critic_loss = F.mse_loss(current_Qs, target_expanded)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) # gradient clipping
        self.critic_opt.step()

        self._soft_update_critic()

        return {'critic_loss': critic_loss.item(), 'target_Q': target_Q.mean().item()}

    def update_actor(self, replay_iter):
        obs, _, _, _, _ = [x.to(self.device) for x in replay_iter]

        dist = self.actor.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        Qs = self.critic.forward(obs, action)
        Qs = torch.stack(Qs, dim=1)
        Q_min = Qs.min(dim=1).values

        actor_loss = (self.alpha * log_prob - Q_min).mean() # entropy regularization
        # actor_loss = -Q_min.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # gradient clipping
        self.actor_opt.step()

        return {'actor_loss': actor_loss.item(), 'entropy': -log_prob.mean().item()}

    def _soft_update_critic(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.critic_tau * param.data + (1.0 - self.critic_tau) * target_param.data
            )
