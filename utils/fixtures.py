import pandas as pd
import requests

def fixture_ctx_for_gw(players_df: pd.DataFrame, player_id: int, gw: int):
    """Return (was_home:int, fdr:int) for player's team in GW or None if no fixture."""
    team_id = int(players_df.loc[players_df['id'] == player_id, 'team'].iloc[0])
    fx = pd.DataFrame(requests.get(f"https://fantasy.premierleague.com/api/fixtures/?event={gw}").json())
    if fx.empty:
        return None
    row = fx[(fx['team_h'] == team_id) | (fx['team_a'] == team_id)]
    if row.empty:
        return None
    row = row.iloc[0]
    was_home = int(row['team_h'] == team_id)
    fdr = int(row['team_h_difficulty'] if was_home else row['team_a_difficulty'])
    return was_home, fdr
