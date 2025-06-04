import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)
        self.hidden = None

    def forward(self, x):
        x = self.fc(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, self.hidden = self.lstm(x, self.hidden)
        x = x.squeeze(1)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

    def reset_hidden(self):
        self.hidden = None


class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        returns, advantages = [], []
        gae = 0
        next_value = 0
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * next_value * (1 - self.dones[step]) - self.values[step]
            gae = delta + gamma * lam * (1 - self.dones[step]) * gae
            next_value = self.values[step]
            returns.insert(0, gae + self.values[step])
            advantages.insert(0, gae)
        return torch.tensor(returns), torch.tensor(advantages)


def ppo_update(model, optimizer, trajectory, writer, global_step, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):
    states = torch.stack(trajectory.states)
    actions = torch.tensor(trajectory.actions)
    old_log_probs = torch.stack(trajectory.log_probs)
    returns, advantages = trajectory.compute_returns_and_advantages()

    model.reset_hidden()
    logits, values = model(states)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values.squeeze(), returns)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    writer.add_scalar("Losses/Total", loss.item(), global_step)
    writer.add_scalar("Losses/Policy", policy_loss.item(), global_step)
    writer.add_scalar("Losses/Value", value_loss.item(), global_step)
    writer.add_scalar("Losses/Entropy", entropy.item(), global_step)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate_agent(env, model, eval_episodes=10):
    model.eval()
    total_reward = 0
    for _ in range(eval_episodes):
        model.reset_hidden()
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, _ = model(state_tensor)
                avail_mask = torch.tensor(state, dtype=torch.bool).unsqueeze(0)
                masked_logits = logits.masked_fill(~avail_mask, -float('inf'))
                probs = torch.softmax(masked_logits, dim=-1)
                num_actions = probs.shape[-1]
                decay = torch.exp(-torch.arange(num_actions, dtype=torch.float32) / 30.0)
                probs = probs * decay
                probs = probs / probs.sum()
                action = torch.argmax(probs, dim=-1)
            state, reward, done, _ = env.step(action.item())
            total_reward += reward
    model.train()
    return total_reward / eval_episodes


def train_ppo(env, model, optimizer, episodes=1000):
    writer = SummaryWriter(log_dir=f"runs/ppo/{time.strftime('%Y%m%d-%H-%M-%S')}")
    global_step = 0

    for episode in range(episodes):
        model.reset_hidden()
        state = env.reset()
        done = False
        traj = Trajectory()
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = model(state_tensor)
            avail_mask = torch.tensor(state, dtype=torch.bool).unsqueeze(0)
            masked_logits = logits.masked_fill(~avail_mask, -float('inf'))
            probs = torch.softmax(masked_logits, dim=-1)
            num_actions = probs.shape[-1]
            decay = torch.exp(-torch.arange(num_actions, dtype=torch.float32) / 30.0)
            probs = probs * decay
            probs = probs / probs.sum()

            dist = Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())

            traj.add(state_tensor.squeeze(), action.item(), log_prob, reward, done, value.item())
            state = next_state
            total_reward += reward
            global_step += 1

        writer.add_scalar("Train/EpisodeReward", total_reward, episode)

        if episode % 50 == 0:
            eval_reward = evaluate_agent(env, model)
            writer.add_scalar("Eval/AvgReward", eval_reward, episode)
            print(f"Episode {episode} | Eval Reward: {eval_reward:.2f}")

        ppo_update(model, optimizer, traj, writer, global_step)
