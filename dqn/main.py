import argparse
import csv
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ac.gym_env import GymEnv
from .trainer import train
import time

def load_players(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FANTASY_BBALL_DRAFT_DQN')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='number of epochs Q-learning is run for')
    parser.add_argument('--random_seed', type=int, default=12,
                        help='manual seed for pytorch')
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    csv_path = 'data/nba_players_stats_2025_finalized.csv'
    players_data = load_players(csv_path)
    env = GymEnv(players_data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    steps_per_episode = env.roster_size

    writer = SummaryWriter(log_dir=f"runs/dqn/{time.strftime('%Y%m%d-%H-%M-%S')}")

    train(
        env=env,
        input_dim=obs_dim,
        action_dim=action_dim,
        num_epochs=args.num_epochs,
        steps_per_episode=steps_per_episode,
        writer=writer,
    )
