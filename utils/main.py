import sys
import pandas as pd

from utils.predictor import predict_gameweek
from utils.select_team import create_team
from utils.config import FEATURES, TARGET
from utils.request_data import request_data

def main(gw: int):
    data = request_data("https://fantasy.premierleague.com/api/bootstrap-static/")
    players = pd.DataFrame(data['elements'])
    players = players.where(players['minutes'] > 300).dropna(subset=['minutes'])
    players["id"] = players["id"].astype(int)

    predictions_df = predict_gameweek(data, players, gw, FEATURES, TARGET, N_RUNS=5)

    out_path = f"data/gw{gw}_predicted_points.csv"
    predictions_df.to_csv(out_path, columns=["player_id","player_name","round","predicted_points"], index=False)
    print(f"Saved predictions to {out_path}")

    create_team(gw, predictions_df, players)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python main.py <GW>")

    gw = int(sys.argv[1])
    main(gw)
