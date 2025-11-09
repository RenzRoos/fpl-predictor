import requests
import pandas as pd
import numpy as np

def get_play_probability(player_meta, gw_requested: int) -> float:
    # Determine current live GW from fixtures API
    fixtures = pd.DataFrame(
        requests.get("https://fantasy.premierleague.com/api/fixtures/").json()
    )
    current_gw = int(fixtures.loc[fixtures["finished"] == True, "event"].max()) + 1

    if gw_requested == current_gw:
        return player_meta["chance_of_playing_this_round"]
    elif gw_requested == current_gw + 1:
        return player_meta["chance_of_playing_next_round"]
    else:
        return np.nan