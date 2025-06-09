import torch
import numpy as np
import csv
import argparse
from .ac import ACAgent
from .gym_env import GymEnv
from .train import train_agent
from .replay_buffer import ReplayBuffer

def load_players(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def main():
    parser = argparse.ArgumentParser('FANTASY_BBALL_DRAFT_DSAC')
    parser.add_argument('--log_suffix', type=str, default='',
                        help='suffix for tensorboard log filename')
    parser.add_argument('--roster_size', type=int, default=13,
                        help='roster size for each team (number of draft rounds)')
    args = parser.parse_args()

    csv_path = 'data/nba_players_stats_2025_finalized.csv'
    players_data = load_players(csv_path)
    env = GymEnv(players_data, roster_size=args.roster_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print('Obs_Dim ', obs_dim, 'Action_Dim', action_dim)
    agent = ACAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=2048, device=device)
    buffer = ReplayBuffer(capacity=150000, obs_dim=obs_dim, device=device)
    win_buffer = ReplayBuffer(capacity=15000, obs_dim=obs_dim, device=device)

    train_agent(env, agent, buffer, win_buffer, episodes=30000, log_suffix=args.log_suffix)

if __name__ == "__main__":
    main()
