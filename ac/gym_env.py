import gym
from gym import spaces
import numpy as np
import copy
from game.fantasy_game import create_teams, score_and_rank_teams, calculate_reward, CATS
from game.reward_values import REWARD_UNAVAILABLE_PICK

class GymEnv(gym.Env): # Fantasy Draft game as gym environment
    def __init__(self, players_stats, num_teams=10, roster_size=13):
        super().__init__()
        self.players_stats_original = players_stats
        self.num_teams = num_teams
        self.roster_size = roster_size

        self.all_player_names = [p['Player'] for p in self.players_stats_original]
        self.action_space = spaces.Discrete(len(self.all_player_names))  # pick index from player list
        # self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.all_player_names),), dtype=np.int8)
        obs_dim = len(self.all_player_names) + (self.num_teams * len(CATS)) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.teams = create_teams(self.num_teams, self.num_teams - 1)
        for i, team in enumerate(self.teams):
            if team.team_type == 'rl_agent':
                self.rl_team_index = i
        self.players = copy.deepcopy(self.players_stats_original)
        self.available = np.ones(len(self.players), dtype=np.int8)
        self._initial_step()

        return self._get_obs()

    def step(self, action):
        round_num = len(self.teams[self.rl_team_index].roster)
        pick_order = self.snake_order(self.rl_team_index, round_num)
        reward = 0

        for team_i in pick_order:
            if (len(self.teams[team_i].roster) >= self.roster_size):
                continue
            # RL agent's turn (should always be first)
            if team_i == self.rl_team_index:
                if self.available[action] == 0:
                    reward = REWARD_UNAVAILABLE_PICK
                    self.top_n_random_pick(team_i, 50)
                else:
                    picked_player = self.players[action]['Player']
                    self.teams[self.rl_team_index].roster.append(picked_player)
                    self.available[action] = 0
            # Simulate other team pick
            else:
                if (len(self.teams[team_i].roster) < 5): self.top_n_random_pick(team_i, 5)
                else: self.top_n_random_pick(team_i, 10)

        # Check if draft is over
        done = self._check_done()
        reward += self._evaluate_team(done)
        if done:
            for team in self.teams:
                assert len(team.roster) == self.roster_size, f"Team {team.name} has too many players ({len(team.roster)})"

        return self._get_obs(), reward, done, {}

    def top_n_random_pick(self, team_i, n=10):
        available_indices = [i for i, a in enumerate(self.available) if a == 1]
        top_n_available = available_indices[:n]
        if not top_n_available:
            return
        pick = np.random.choice(top_n_available)
        self.available[pick] = 0
        self.teams[team_i].roster.append(self.players[pick]['Player'])

    def snake_order(self, i, round_number):
        if round_number % 2 == 0:
            sequence = list(range(i, self.num_teams)) + list(reversed(range(i + 1, self.num_teams)))
        else:
            sequence = list(reversed(range(i + 1))) + list(range(0, i))
        return sequence

    def _initial_step(self):
        if (len(self.teams[0].roster)):
            return self._get_obs(), 0, False, {}
        pick_order = list(range(self.num_teams))
        for team_i in pick_order:
            if team_i == self.rl_team_index:
                break
            else:
                self.top_n_random_pick(team_i, 5)

    def _check_done(self):
        # End when RL team has a full roster
        return len(self.teams[self.rl_team_index].roster) >= self.roster_size

    def _get_obs(self):
        league_stats_obs = self._get_league_stats_obs()
        return np.concatenate([self.available.copy().astype(np.float32), league_stats_obs])

    def _evaluate_team(self, done=False):
        players_dict = {p['Player']: p for p in self.players_stats_original}
        ranked = score_and_rank_teams(self.teams, players_dict)
        return calculate_reward(ranked, self.teams[self.rl_team_index].name, done)
    
    def _get_rl_team_roster(self):
        return self.teams[self.rl_team_index].roster

    def _get_league_stats_obs(self):
        if not self.teams[self.rl_team_index].scored_total_stats:
            return np.zeros((len(self.teams) * len(CATS)) + 1, dtype=np.float32)
        
        stats = np.array([self.teams[self.rl_team_index].scored_total_stats[cat] for cat in CATS], dtype=np.float32)
        
        for i, team in enumerate(self.teams):
            if i != self.rl_team_index:
                team_stats = np.array([team.scored_total_stats[cat] for cat in CATS])
                stats = np.concatenate([stats, team_stats], dtype=np.float32)
                
        players_dict = {p['Player']: p for p in self.players_stats_original}
        ranked = score_and_rank_teams(self.teams, players_dict)
        rank_idx = 0
        for i, team in enumerate(ranked):
            if team.name == self.teams[self.rl_team_index].name:
                rank_idx = i
                break
        
        return np.concatenate([stats, np.array([float(rank_idx)])], dtype=np.float32)
