import pandas as pd
import sys 

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

def scout_get_data(gw: int, evaluate: bool = False):
    # load raw scout picks (names-only or with minimal columns)
    scout_path = f"scout_picks/gw{gw}_scout_picks.csv"
    scout_raw = pd.read_csv(scout_path)

    # load global predictions (produced by main.py)
    pred_path = f"data/gw{gw}_predicted_points.csv"
    preds = pd.read_csv(pred_path)[["player_id", "player_name", "round", "predicted_points"]]

    # merge by player_name (scout names must match web_name in preds)
    merged = scout_raw.merge(
        preds,
        on="player_name",
        how="left"
    )

    merged.to_csv(scout_path, index=False)
    print(f"Updated {scout_path} with player_id, round and predicted_points.")

    if evaluate:
        evaluate_scout_picks(gw)


        
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "-e":
        raise SystemExit("Usage: scout_get_data.py <GW> -e")
    gw = int(sys.argv[1])

    evaluate = False
    if len(sys.argv) > 2:
        if sys.argv[2] == "-e":
            evaluate = True

    scout_get_data(gw, evaluate)
