import sys
import pandas as pd
import requests

from predictor import predict_gameweek
from select_team import create_team
from config import FEATURES, TARGET

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python main.py <GW>")

    GW = int(sys.argv[1])

    data = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    players = pd.DataFrame(data['elements'])

    predictions_df = predict_gameweek(data, players, GW, FEATURES, TARGET, N_RUNS=5)

    out_path = f"data/gw{GW}_predicted_points.csv"
    predictions_df.to_csv(out_path, columns=["player_id","player_name","round","predicted_points"], index=False)
    print(f"Saved predictions to {out_path}")

    create_team(GW, predictions_df, players)

if __name__ == "__main__":
    main()
