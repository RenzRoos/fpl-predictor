import pandas as pd
import numpy as np

from utils.config import PLAYER_TARGETS

def compute_predicted_points(preds: pd.DataFrame, players: pd.DataFrame, team_gc: pd.Series ) -> pd.Series:
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    meta = players[["id", "element_type", "team"]].rename(
        columns={"id": "player_id", "team": "team_id"}
    )
    df = preds.merge(meta, on="player_id", how="left")
    df["position"] = df["element_type"].map(pos_map)

    # make sure predicted columns exist
    for t in PLAYER_TARGETS:
        col = f"predicted_{t}"
        if col not in df.columns:
            df[col] = 0.0

    mins   = df["predicted_minutes"]
    goals  = df["predicted_goals_scored"]
    asts   = df["predicted_assists"]
    yc     = df["predicted_yellow_cards"]
    bps    = df["predicted_bps"]
    saves  = df["predicted_saves"]
    pos    = df["position"]

    # minutes points (rough approximation)
    mins_pts = (
        (mins >= 60).astype(float) * 2.0 +
        ((mins >= 30) & (mins < 60)).astype(float) * 1.0 +
        ((mins > 0) & (mins < 30)).astype(float) * 0.5
    )

    # goals points by position
    goal_pts = (
        (pos == "GK").astype(float) * 6 +
        (pos == "DEF").astype(float) * 6 +
        (pos == "MID").astype(float) * 5 +
        (pos == "FWD").astype(float) * 4
    ) * goals

    # assists: 3 pts each
    ast_pts = 3.0 * asts

    # yellow cards: -1 each
    yc_pts = -1.0 * yc

    # BPS: rough scaling
    bps_pts = 0.05 * bps

    # saves: GK 1pt per 3 saves
    save_pts = ((pos == "GK").astype(float) * (saves / 3.0))

    # ---------- NEW: team-level goals_conceded impact ----------

    # map team_id -> predicted goals conceded for that GW
    df["predicted_goals_conceded_team"] = df["team_id"].map(team_gc).fillna(1.5)
    gc = df["predicted_goals_conceded_team"]

    # approximate clean-sheet probability via Poisson(λ=gc): P(CS) ≈ e^{-λ}
    cs_prob = np.exp(-gc)

    cs_base = (
        ((pos == "GK") | (pos == "DEF")).astype(float) * 4.0 +  # GK/DEF clean sheet
        (pos == "MID").astype(float) * 1.0                       # MID clean sheet
        # FWD: 0
    )
    cs_pts = cs_prob * cs_base

    # goals conceded penalty: GK/DEF get -1 per 2 GC → ~ -0.5 * λ
    gc_penalty = ((pos == "GK") | (pos == "DEF")).astype(float) * (-0.5 * gc)

    # -----------------------------------------------------------

    total = mins_pts + goal_pts + ast_pts + yc_pts + bps_pts + save_pts + cs_pts + gc_penalty
    return total
