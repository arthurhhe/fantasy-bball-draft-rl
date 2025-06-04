import torch
import numpy as np
import csv
from .ac import ACAgent
from .gym_env import GymEnv
from .train import train_agent
from .replay_buffer import ReplayBuffer

def load_players(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def main():
    csv_path = 'data/nba_players_stats_2025_finalized.csv'
    players_data = load_players(csv_path)
    env = GymEnv(players_data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print('Obs_Dim ', obs_dim, 'Action_Dim', action_dim)
    agent = ACAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, device=device)
    buffer = ReplayBuffer(capacity=150000, obs_dim=obs_dim, device=device)
    win_buffer = ReplayBuffer(capacity=15000, obs_dim=obs_dim, device=device)

    train_agent(env, agent, buffer, win_buffer, episodes=80000)

if __name__ == "__main__":
    main()
