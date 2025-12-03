import requests
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=None)
def get_fdr(gameweek: int, opponent: str, home: bool) -> int:
    """
    Return the FPL difficulty rating (FDR) for a given opponent in a specific gameweek.

    Parameters
    ----------
    gameweek : int
        The gameweek number (1-38).
    opponent : str
        Opponent team name (case-insensitive, e.g. 'Tottenham Hotspur').
    home : bool
        True if your team is playing at home, False if away.

    Returns
    -------
    int
        The FDR score (1-5), or None if not found.
    """

    # Load team metadata for name <-> id mapping
    bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name"]]
    name_to_id = {n.lower(): i for i, n in zip(teams["id"], teams["name"])}

    # Standardize opponent name
    opp_name = opponent.lower()
    if opp_name not in name_to_id:
        raise ValueError(f"Opponent '{opponent}' not found in FPL teams list.")

    opp_id = name_to_id[opp_name]

    # Get fixtures for the specified gameweek
    fixtures = pd.DataFrame(requests.get(f"https://fantasy.premierleague.com/api/fixtures/?event={gameweek}").json())
    if fixtures.empty:
        return None

    # Select the correct FDR field based on home/away perspective
    if home:
        fdr_row = fixtures.loc[fixtures["team_a"] == opp_id, "team_h_difficulty"]
    else:
        fdr_row = fixtures.loc[fixtures["team_h"] == opp_id, "team_a_difficulty"]

    # Return FDR if found
    return int(fdr_row.iloc[0]) if not fdr_row.empty else None

