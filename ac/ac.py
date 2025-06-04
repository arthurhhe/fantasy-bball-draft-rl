import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def build_net(layer_shape, hidden_activation, output_activation):
	layers = []
	for j in range(len(layer_shape) - 1):
		act = hidden_activation if j < len(layer_shape) - 2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
	return nn.Sequential(*layers)

class Double_Q_Net(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim):
		super(Double_Q_Net, self).__init__()
		layers = [obs_dim] + [hidden_dim, hidden_dim] + [action_dim]

		self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		q1 = self.Q1(s)
		q2 = self.Q2(s)
		return q1,q2

class Policy_Net(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim):
		super(Policy_Net, self).__init__()
		layers = [obs_dim] + [hidden_dim, hidden_dim] + [action_dim]
		self.P = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		logits = self.P(s)
		probs = F.softmax(logits, dim=1)
		return probs

# class Actor(nn.Module):
#     def __init__(self, obs_dim, action_dim, hidden_dim):
#         super(Actor, self).__init__()
#         self.policy = nn.Sequential(
#             nn.Linear(obs_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, action_dim)
#         )

#     def forward(self, obs):
#         logits = self.policy(obs)
#         dist = Categorical(logits=logits)
#         return dist

# class Critic(nn.Module):
#     def __init__(self, obs_dim, action_dim, num_critics, hidden_dim):
#         super(Critic, self).__init__()
#         self.action_dim = action_dim
        
#         self.critics = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(obs_dim + action_dim, hidden_dim),
#                 nn.LayerNorm(hidden_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.LayerNorm(hidden_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim, 1)
#             ) for _ in range(num_critics)
#         ])

#     def forward(self, obs, action):
#         if action.dim() == 1:
#             action = F.one_hot(action, num_classes=self.action_dim).float()
#         elif action.shape[-1] != self.action_dim:
#             raise ValueError("Expected action to be one-hot or indices.")

#         h_action = torch.cat([obs, action], dim=-1)
#         return [critic(h_action) for critic in self.critics]

class ACAgent:
    def __init__(self, obs_dim, action_dim, hidden_dim, device, lr=2e-5, critic_tau=0.005, gamma=0.99, alpha=0.3):
        self.device = device
        self.gamma = gamma
        self.critic_tau = critic_tau
        self.alpha = alpha
        self.action_dim = action_dim

        # self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor = Policy_Net(obs_dim, action_dim, hidden_dim).to(device)
        # self.critic = Critic(obs_dim, action_dim, num_critics, hidden_dim).to(device)
        self.critic = Double_Q_Net(obs_dim, action_dim, hidden_dim).to(device)
        # self.critic_target = Critic(obs_dim, action_dim, num_critics, hidden_dim).to(device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters(): p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, obs, eval_mode=False, early_bias=True):
        with torch.no_grad():
            obs_array = np.array(obs)
            obs = torch.FloatTensor(obs[np.newaxis,:]).to(self.device) # from (obs_dim,) to (1, obs_dim)
            probs = self.actor(obs)
            
            availability_mask = torch.tensor(obs_array[:self.action_dim], dtype=torch.float32, device=self.device)
            probs = probs * availability_mask

            if early_bias:
                num_actions = probs.shape[-1]
                decay = torch.exp(-torch.arange(num_actions, device=self.device, dtype=torch.float32) / 30.0) # decay factor
                probs = probs * decay
                probs = probs / probs.sum()

            # if eval_mode:
            #     action = probs.argmax(dim=-1).item()
            # else:
            action = Categorical(probs).sample().item()
            return action

    def update_critic(self, replay_iter):
        obs, action, reward, next_obs, done = [x.to(self.device) for x in replay_iter] # maybe check if this is right

        with torch.no_grad():
            next_probs = self.actor(next_obs) #[batch_size, action_dim]
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1_all, next_q2_all = self.critic_target(next_obs)
            min_next_q_all = torch.min(next_q1_all, next_q2_all)
            v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True) # [batch_size, 1]
            target_Q = reward + (~done) * self.gamma * v_next

        q1_all, q2_all = self.critic(obs) #[batch_size, action_dim]
        assert q1_all.shape[1] == self.action_dim and q2_all.shape[1] == self.action_dim, f"Q Shape Mismatch | q1_all shape: {q1_all.shape} | q2_all shape: {q2_all.shape} | action_dim: {self.action_dim}"
        q1, q2 = q1_all.gather(1, action), q2_all.gather(1, action) #[batch_size, 1]
        critic_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        self._soft_update_critic()

        return {'critic_loss': critic_loss.item(), 'target_Q': target_Q.mean().item()}

    def update_actor(self, replay_iter):
        obs, _, _, _, _ = [x.to(self.device) for x in replay_iter]
        probs = self.actor(obs) #[batch_size, action_dim]
        assert probs.shape[1] == self.action_dim, f"Actor probs shape unexpected: {probs.shape}"
        log_probs = torch.log(probs + 1e-8)

        with torch.no_grad():
            q1_all, q2_all = self.critic(obs)  #[batch_size, action_dim]
        min_q_all = torch.min(q1_all, q2_all)

        actor_loss = torch.sum(probs * (self.alpha * log_probs - min_q_all), dim=1, keepdim=False).mean() #[batch_size,]

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        return {'actor_loss': actor_loss.item(), 'entropy': -log_probs.mean().item()}

    def _soft_update_critic(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.critic_tau * param.data + (1.0 - self.critic_tau) * target_param.data
            )
