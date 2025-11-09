import pandas as pd
import numpy as np
from utils.fixtures import fixture_ctx_for_gw

def build_X_pred_for_gw(mh: pd.DataFrame, players_df: pd.DataFrame,
                        player_id: int, gw: int, FEATURES: list, TARGET: str):
    """Create a single feature row for GW using only rounds < GW + fixture context for GW."""
    past = mh[mh['round'] != gw].copy()
    if past.empty:
        return None

    past['status_played'] = (past['status'] == 'Played').astype(int)
    past['was_home'] = past['was_home'].astype(int)
    for c in FEATURES + [TARGET]:
        if c not in past.columns:
            past[c] = np.nan
    past[FEATURES + [TARGET]] = past[FEATURES + [TARGET]].apply(pd.to_numeric, errors='coerce')

    means = past[FEATURES].mean(numeric_only=True).to_frame().T

    ctx = fixture_ctx_for_gw(players_df, player_id, gw)
    if ctx is None:
        return None
    was_home, fdr = ctx
    means['was_home'] = float(was_home)
    means['fdr_score'] = float(fdr)

    return means[FEATURES]
