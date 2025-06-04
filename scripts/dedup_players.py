import csv
PLAYER_STATS_CSV_PATH = 'data/nba_players_stats_2025_working.csv'

def save_players_to_csv(players, season):
    filename=f"nba_players_stats_{season}_deduped.csv"
    if not players:
        print("No player data to save.")
        return

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=players[0].keys())
        writer.writeheader()
        writer.writerows(players)

def dedup_players_csv():
    players = []
    with open(PLAYER_STATS_CSV_PATH, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            players.append(row)
    print('# Rows', len(players))
    players_dict = { p['Player']: [] for p in players }
    for p in players:
        players_dict[p['Player']].append(p)
    unique_players = {p: entries[0] for p, entries in players_dict.items() if len(entries) == 1}
    duplicate_players = {p: entries for p, entries in players_dict.items() if len(entries) > 1}
    print('# Duplicates', len(duplicate_players))
    deduped_players = {}
    fields_to_avg = ['MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'eFG%', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    for p, entries in duplicate_players.items():
        player_entry = { 'Player': p, 'Rk': 0, 'TEAM': entries[0]['TEAM'], 'Age': entries[0]['Age'], 'Pos': entries[0]['Pos'],}
        total_gp, total_gs = 0, 0
        for e in entries:
            total_gp += int(e['G'])
            total_gs += int(e['GS'])
            if e['PADP']: player_entry['PADP'] = e['PADP']
            if e['DADP']: player_entry['DADP'] = e['DADP']
        player_entry['G'] = total_gp
        player_entry['GS'] = total_gs
        for f in fields_to_avg:
            player_entry[f] = round(sum((float(e[f])*int(e['G'])/total_gp) for e in entries), 1) if e[f] != '' else ''
        # percentages_to_calc = ['FG%', '3P%', '2P%', 'FT%']
        player_entry['FG%'] = round((player_entry['3P'] + player_entry['2P'])/(player_entry['3PA'] + player_entry['2PA']), 3)
        player_entry['FT%'] = round(player_entry['FT'] / player_entry['FTA'], 3) if player_entry['FTA'] else ''
        player_entry['3P%'] = round(player_entry['3P'] / player_entry['3PA'], 3) if player_entry['3PA'] else ''
        player_entry['2P%'] = round(player_entry['2P'] / player_entry['2PA'], 3) if player_entry['2PA'] else ''
        deduped_players[p] = player_entry
    print('# Uniques', len(unique_players.keys()))
    unique_players.update(deduped_players)
    print('# Combined', len(unique_players.keys()))
    player_list = [value for value in unique_players.values()]
    player_list.sort(key=lambda p: float(p.get('PADP', 999) if p.get('PADP', 999) else 999), reverse=False)
    save_players_to_csv(player_list, 2025)

dedup_players_csv()