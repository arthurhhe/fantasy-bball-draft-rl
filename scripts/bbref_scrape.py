import time
import random
import csv
import requests
from bs4 import BeautifulSoup
from game.fantasy_game import SELECTED_SEASON_END_YEAR

nba_teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64)",
]

def scrape_per_game_player_stats(team, season):
    url = f"https://www.basketball-reference.com/teams/{team}/{season}.html"
    print('Scraping', team, season)
    req_headers = {"User-Agent": random.choice(USER_AGENTS)}
    res = requests.get(url, headers=req_headers)
    print('Status', res.status_code)  # Should be 200

    soup = BeautifulSoup(res.content, 'html.parser')
    table = soup.find('table', id='per_game_stats')

    headers = [th.get_text(strip=True) for th in table.find('thead').find_all('th')]

    players = []
    for row in table.find('tbody').find_all('tr'):
        cells = row.find_all(['th', 'td'])
        player_data = {headers[i]: cells[i].get_text(strip=True) for i in range(len(cells))}
        player_data['TEAM'] = team
        players.append(player_data)

    return players

def save_players_to_csv(players, season):
    filename=f"nba_players_stats_{season}.csv"
    if not players:
        print("No player data to save.")
        return

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=players[0].keys())
        writer.writeheader()
        writer.writerows(players)

def scrape_all_players_for_season(season):
    players = []
    for team in nba_teams:
        team_players = scrape_per_game_player_stats(team, season)
        players += team_players
        delay = random.uniform(3, 6) # BBRef rate limits
        print(f"Sleeping for {delay:.2f} seconds...")
        time.sleep(delay)
    save_players_to_csv(players, season)
    return players

scrape_all_players_for_season(SELECTED_SEASON_END_YEAR)
