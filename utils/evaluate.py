from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_points_scored import update_actual_points
from select_team import evaluate_team_performance

def evaluate_predictions(gameweek: int):
    """Evaluate model accuracy for a given gameweek."""
    csv_path = f"data/gw{gameweek}_predicted_points.csv"
    df = pd.read_csv(csv_path)

    mask = ~((df["predicted_points"] == 0) & (df["actual_points"] == 0))
    df = df.loc[mask].copy()

    # clean data
    df = df.dropna(subset=["predicted_points", "actual_points"])
    y_true = df["actual_points"]
    y_pred = df["predicted_points"]

    # core metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"Evaluation for Gameweek {gameweek}")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")



if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <gameweek>")
        sys.exit(1)

    GW = int(sys.argv[1])

    update_actual_points(GW)
    evaluate_team_performance(GW)
    evaluate_predictions(GW)