import torch
import numpy as np
import csv
from ac.gym_env import GymEnv
from .train import PPOActorCritic, train_ppo
from .lstm_train import PPOActorCritic as LSTMActorCritic, train_ppo as train_lstm

def load_players(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def main():
    csv_path = 'data/nba_players_stats_2025_finalized.csv'
    players_stats = load_players(csv_path)
    env = GymEnv(players_stats)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = PPOActorCritic(obs_dim, act_dim)
    # model = LSTMActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    train_ppo(env, model, optimizer, episodes=10000)
    # train_lstm(env, model, optimizer, episodes=20000)

if __name__ == "__main__":
    main()
