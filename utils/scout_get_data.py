import requests
import pandas as pd
import sys 

from config import FEATURES, TARGET
from predictor import predict_gameweek

def evaluate_scout_picks(gw: int):
    squad_path = f"scout_picks/gw{gw}_scout_picks.csv"
    df = pd.read_csv(squad_path)

    pred_path = f"data/gw{gw}_predicted_points.csv"
    all_players = pd.read_csv(pred_path)[["player_id", "actual_points"]]

    team = df.merge(all_players, on="player_id", how="left")
    team["actual_points"] = team["actual_points"].fillna(0)

    total_actual_points = team["actual_points"].sum()
    total_predicted_points = team["predicted_points"].fillna(0).sum()

    print(f"Total predicted points for GW{gw} squad: {total_predicted_points:.1f}")
    print(f"Total actual points for GW{gw} squad: {total_actual_points:.1f}")
    print(f"Difference: {total_actual_points - total_predicted_points:.1f}")

    team.to_csv(squad_path, index=False)
    print(f"Updated {squad_path} with actual_points column.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "-e":
        raise SystemExit("Usage: scout_get_data.py <GW> -e")
    gw = int(sys.argv[1])

    evaluate = False
    if len(sys.argv) > 2:
        if sys.argv[2] == "-e":
            evaluate = True


    data = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    players = pd.DataFrame(data["elements"])

    scout_picks = pd.read_csv(f"scout_picks/gw{gw}_scout_picks.csv")
    scout_picks = players[players["web_name"].isin(scout_picks["player_name"])].copy()

    predictions_df = predict_gameweek(data, scout_picks, gw, FEATURES, TARGET, N_RUNS=5)

    predictions_df.to_csv(f"scout_picks/gw{gw}_scout_picks.csv", index=False)
    print(f"Updated scout_picks/gw{gw}_scout_picks.csv with predicted points and player IDs.")

    if evaluate:
        evaluate_scout_picks(gw)