import sys
import pandas as pd
import requests

from utils.predictor import predict_gameweek
from utils.select_team import create_team
from utils.config import FEATURES, TARGET

def main(gw: int):
    data = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    players = pd.DataFrame(data['elements'])

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
