import csv
from game.fantasy_game import score_and_rank_teams
from game.fantasy_game import Team

def load_players(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def main():
    csv_path = 'data/nba_players_stats_2025_finalized.csv'
    players_stats = load_players(csv_path)
    
    teams = [Team(f"Team {p['Player']}", team_type='experiment', roster=[p['Player']]) for p in players_stats]
    players_dict = {p['Player']: p for p in players_stats}

    teams_ranked = score_and_rank_teams(teams, players_dict)

    for team in teams_ranked[:10]:
        print(f"{team.roster} {team.scored_win_percent}")

if __name__ == "__main__":
    main()

# RESULTS
# ['Shai Gilgeous-Alexander'] 0.8317683881064163
# ['Nikola Jokić'] 0.8129890453834115
# ['Tyrese Haliburton'] 0.7873630672926447
# ['Karl-Anthony Towns'] 0.7848200312989045
# ['Desmond Bane'] 0.7779733959311425
# ['Christian Braun'] 0.776017214397496
# ['Cade Cunningham'] 0.7740610328638498
# ['Anthony Edwards'] 0.7723004694835681
# ['Kevin Durant'] 0.770735524256651
# ['LeBron James'] 0.7697574334898278