import pandas as pd
import random
import csv
from .reward_values import REWARD_UNFOUND, REWARD_FIRST_PLACE_PARTIAL, REWARD_SECOND_PLACE_PARTIAL, REWARD_THIRD_PLACE_PARTIAL, REWARD_FIRST_PLACE_FINAL, REWARD_SECOND_PLACE_FINAL, REWARD_THIRD_PLACE_FINAL

# Constants
NUM_TEAMS = 10
ROSTER_SPOTS = ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL' 'UTIL', 'UTIL', 'BENCH', 'BENCH', 'BENCH']
CATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FG%', 'FT%', 'TO']
PLAYER_CATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FGA', 'FGM', 'FTA', 'FTM', 'TO', 'GP']
BBREF_PLAYER_CATS = {
    'PTS': 'PTS',
    'REB': 'TRB',
    'AST': 'AST',
    'STL': 'STL',
    'BLK': 'BLK',
    '3PM': '3P',
    'FGA': 'FGA',
    'FGM': 'FG',
    'FTA': 'FTA',
    'FTM': 'FT',
    'TO': 'TOV',
    'GP': 'G',
}
TEAM_SCORING_CATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'FGA', 'FGM', 'FTA', 'FTM', 'TO']
GP_MAX = 82
SELECTED_SEASON_END_YEAR = 2025
PLAYER_STATS_CSV_PATH = 'data/nba_players_stats_2025_finalized.csv'

# class Player:
#     def __init__(self, name):
#         self.name = name
#         self.season_averages = {}
#         self.game_history = []

#     def __str__(self):
#         return f"{self.name}"

class Team:
    def __init__(self, name, team_type='random', roster=None):
        self.name = name
        self.team_type = team_type
        self.roster = roster if roster is not None else []
        self.draft_position = -1
        self.scored_total_stats = None
        self.scored_wins = 0
        self.scored_losses = 0
        self.scored_ties = 0
        self.scored_win_percent = -1

    def __str__(self):
        return f"{self.name}"

def create_teams(num_teams, num_random_bot_teams):
    teams = [Team(f"Team {i} (R)") for i in range(1, num_random_bot_teams+1)]
    teams += [Team(f"Team {i} (RLA)", "rl_agent") for i in range(num_random_bot_teams+1, num_teams+1)]
    random.shuffle(teams)
    return teams

# TODO: get rl agent pick
def poll_rl_agent_pick():
    return random.randint(0, 5)

# Snake draft simulation
def snake_draft(players, num_teams, roster_size):
    teams = create_teams(num_teams, num_teams - 1)
    draft_order = list(range(num_teams))
    
    for round_num in range(roster_size):
        if round_num % 2 == 0:
            order = draft_order
        else:
            order = draft_order[::-1]
        
        for team_i in order:
            if len(players) < 1: break
            pick = 0
            if teams[team_i].team_type == 'random':
                pick = random.randint(0, 9)
            elif teams[team_i].team_type == 'rl_agent':
                pick = poll_rl_agent_pick()
            teams[team_i].roster.append(players.pop(pick)['Player'])

    return teams

def score_and_rank_teams(teams, players_dict):
    season = SELECTED_SEASON_END_YEAR
    for team in teams:
        totals = {cat: 0 for cat in TEAM_SCORING_CATS}
        totals['GP'], totals['FG%'], totals['FT%'] = 0, 0, 0
        for player_name in team.roster:
            player = players_dict[player_name]
            normalizer = int(player['G']) / GP_MAX
            for cat in TEAM_SCORING_CATS:
                totals[cat] += (float(player[BBREF_PLAYER_CATS[cat]]) * normalizer)
            totals['GP'] += int(player['G'])
        totals['FG%'] = totals['FGM'] / totals['FGA'] if totals['FGA'] > 0 else 0
        totals['FT%'] = totals['FTM'] / totals['FTA'] if totals['FTA'] > 0 else 0
        team.scored_total_stats = totals

    # Tally based on matchups
    for cat in CATS:
        for team_A in teams:
            if cat != 'TO':
                team_A.scored_wins += sum(1 if team_A.scored_total_stats[cat] > team_B.scored_total_stats[cat] else 0 for team_B in teams if team_A.name != team_B.name)
                team_A.scored_losses += sum(1 if team_A.scored_total_stats[cat] < team_B.scored_total_stats[cat] else 0 for team_B in teams if team_A.name != team_B.name)
            else:
                team_A.scored_wins += sum(1 if team_A.scored_total_stats[cat] < team_B.scored_total_stats[cat] else 0 for team_B in teams if team_A.name != team_B.name)
                team_A.scored_losses += sum(1 if team_A.scored_total_stats[cat] > team_B.scored_total_stats[cat] else 0 for team_B in teams if team_A.name != team_B.name)
            team_A.scored_ties += sum(1 if team_A.scored_total_stats[cat] == team_B.scored_total_stats[cat] else 0 for team_B in teams if team_A.name != team_B.name)
    
    for team in teams:
        team.scored_win_percent = team.scored_wins / (team.scored_wins + team.scored_losses + team.scored_ties)
    
    teams_ranked = sorted(teams, key=lambda t: t.scored_win_percent, reverse=True)
    return teams_ranked

def calculate_reward(ranked_teams, team_name, final):
    index = -1
    for i, team in enumerate(ranked_teams):
        if team.name == team_name:
            index = i
    # if index < 3 and final: print(f"In {index + 1} place: ", ranked_teams[index].roster)
    if index == -1: return REWARD_UNFOUND
    if index == 0: return REWARD_FIRST_PLACE_FINAL if final else REWARD_FIRST_PLACE_PARTIAL
    if index == 1: return REWARD_SECOND_PLACE_FINAL if final else REWARD_SECOND_PLACE_PARTIAL
    if index == 2: return REWARD_THIRD_PLACE_FINAL if final else REWARD_THIRD_PLACE_PARTIAL
    return REWARD_UNFOUND

# Simulation
def run_simulation():
    # players =  pd.read_csv(PLAYER_STATS_CSV_PATH)
    players = []
    with open(PLAYER_STATS_CSV_PATH, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            players.append(row)
    players_dict = { p['Player']: p for p in players }

    teams = snake_draft(players, NUM_TEAMS, len(ROSTER_SPOTS))
    teams_ranked = score_and_rank_teams(teams, players_dict)

    for _, team in enumerate(teams_ranked):
        print(f"{team.name}: Winning Pct = {team.scored_win_percent:.3f}")
        print(', '.join(team.roster))

if __name__ == "__main__":
    run_simulation()
