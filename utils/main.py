import sys
import pandas as pd

from utils.predictor import predict_gameweek
from utils.select_team import create_team
from utils.config import FEATURES, PLAYER_TARGETS
from utils.compute_points import compute_predicted_points
from utils.estimate_goals_conceded import estimate_team_goals_conceded
from utils.request_data import request_data

def main(gw: int):
    data = request_data("https://fantasy.premierleague.com/api/bootstrap-static/")
    players = pd.DataFrame(data["elements"])

    players = players[players["minutes"] > 450]
    players = players[
        (players["chance_of_playing_next_round"] >= 50) |
        (players["chance_of_playing_next_round"].isna())
    ]
    
    all_preds = None

    # run a separate model per player-level target, then merge
    for target in PLAYER_TARGETS:
        df_t = predict_gameweek(data, players, gw, FEATURES, target, N_RUNS=5)
        colname = f"predicted_{target}"

        if all_preds is None:
            all_preds = df_t[["player_id", "player_name", "round", colname]].copy()
        else:
            all_preds = all_preds.merge(
                df_t[["player_id", colname]],
                on="player_id",
                how="left"
            )

    if all_preds is None or all_preds.empty:
        raise SystemExit(f"No predictions produced for GW{gw}")

    # ---------- NEW: predict goals_conceded per team once ----------
    
    team_ga = estimate_team_goals_conceded(data, gw)  # Series indexed by team_id

    # ---------- use team_gc in per-player points ----------

    all_preds = all_preds.join(compute_predicted_points(all_preds, players, team_ga),  rsuffix="_pts")

    all_preds = all_preds.sort_values(by="predicted_points", ascending=False)

    out_path = f"data/predicted_all/gw{gw}_predicted_points.csv"
    all_preds.to_csv(
        out_path,
        columns=["player_id", "player_name", "predicted_points", "predicted_minutes", "predicted_goals_scored",
                 "predicted_assists", "predicted_yellow_cards", "predicted_bonus", "predicted_saves",
                 "predicted_clean_sheet", "predicted_goals_conceded"],
        index=False
    )
    print(f"Saved predictions to {out_path}")

    create_team(gw, all_preds, players)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python main.py <GW>")

    gw = int(sys.argv[1])
    main(gw)
