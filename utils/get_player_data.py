import pandas as pd
import requests
import numpy as np
from utils.fdr_score import get_fdr

def get_player_match_history(data: pd.DataFrame, player_id: int) -> pd.DataFrame:
    # Player -> team id and name maps
    bootstrap = data
    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id","name"]]
    team_map = dict(zip(teams["id"], teams["name"]))
    team_id = int(players.loc[players["id"] == player_id, "team"].iloc[0])

    # Fixtures -> only finished
    fixtures = pd.DataFrame(requests.get("https://fantasy.premierleague.com/api/fixtures/").json())
    fixtures = fixtures.dropna(subset=["event"])
    played = fixtures[(fixtures["finished"] == True) | (fixtures["finished_provisional"] == True)]

    # Team-only played fixtures with opponent + H/A + score context
    team_fix = played[(played["team_h"] == team_id) | (played["team_a"] == team_id)].copy()
    team_fix["round"] = team_fix["event"].astype(int)
    team_fix["was_home"] = team_fix["team_h"].eq(team_id)
    team_fix["opponent_team"] = np.where(team_fix["was_home"], team_fix["team_a"], team_fix["team_h"]).astype(int)
    team_fix["team_h_name"] = team_fix["team_h"].map(team_map)
    team_fix["team_a_name"] = team_fix["team_a"].map(team_map)
    team_fix["opponent_name"] = team_fix["opponent_team"].map(team_map)

    team_fix = team_fix[[
        "round","was_home","opponent_team","opponent_name",
        "team_h","team_a","team_h_name","team_a_name",
        "team_h_score","team_a_score","kickoff_time"
    ]].sort_values(["round","kickoff_time"]).reset_index(drop=True)

    # Player per-match history (only rounds where he played > 0 mins are present)
    hist_json = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{player_id}/").json()
    df_hist = pd.DataFrame(hist_json.get("history", []))

    # Broad column wishlist (many exist; some may not in older seasons)
    candidate_cols = [
        "round","minutes",
        "goals_scored","assists",
        "goals_conceded","clean_sheets",
        "own_goals","penalties_saved","penalties_missed",
        "yellow_cards","red_cards",
        "saves","bonus","bps",
        "influence","creativity","threat","ict_index",
        "expected_goals","expected_assists",
        "expected_goal_involvements","expected_goals_conceded",
        "was_home","opponent_team","kickoff_time",
        "team_h_score","team_a_score","total_points"
    ]

    # Keep only columns that exist this season
    keep_cols = [c for c in candidate_cols if c in df_hist.columns]
    if df_hist.empty:
        df_hist = pd.DataFrame(columns=keep_cols)
    else:
        df_hist = df_hist[keep_cols].copy()

    # Ensure integer round
    if "round" in df_hist.columns:
        df_hist["round"] = df_hist["round"].astype(int)

    # Merge: only played team fixtures; missing history => DNP
    full = team_fix.merge(df_hist.drop(columns=[c for c in ["was_home","opponent_team","kickoff_time","team_h_score","team_a_score"] if c in df_hist.columns]),
                        on="round", how="left")

    # Label and fill numeric match stats for DNPs
    full["status"] = np.where(full["minutes"].isna(), "DNP", "Played")

    # Define which columns are numeric match stats (intersect with keep_cols)
    numeric_stat_candidates = [
        "minutes","goals_scored","assists","goals_conceded","clean_sheets",
        "own_goals","penalties_saved","penalties_missed",
        "yellow_cards","red_cards","saves","bonus","bps",
        "influence","creativity","threat","ict_index",
        "expected_goals","expected_assists","expected_goal_involvements","expected_goals_conceded",
        "total_points"
    ]
    num_cols = [c for c in numeric_stat_candidates if c in full.columns]

    # Convert to numeric where present, then fill 0 for DNPs
    for c in num_cols:
        full[c] = pd.to_numeric(full[c], errors="coerce")
    full.loc[full["status"]=="DNP", num_cols] = full.loc[full["status"]=="DNP", num_cols].fillna(0)

    # Helpful identifiers
    full["player_id"] = player_id
    full["fdr_score"] = full.apply(
        lambda r: get_fdr(data, int(r["round"]), str(r["opponent_name"]), bool(r["was_home"])),
        axis=1
    )

    # Reorder for readability
    show_cols = ([
        "round","was_home","opponent_name","status",
        "minutes","total_points","team_h_score","team_a_score", "fdr_score"
    ] + [c for c in num_cols if c not in ["minutes","total_points"]])
    show_cols = [c for c in show_cols if c in full.columns]

    return full[show_cols].sort_values(["round","was_home"]).reset_index(drop=True)
