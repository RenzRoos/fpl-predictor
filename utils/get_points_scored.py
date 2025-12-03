import pandas as pd
import requests
import sys

def update_actual_points(gameweek: int):
    csv_path = f"data/gw{gameweek}_predicted_points.csv"

    df = pd.read_csv(csv_path)
    updated = []

    for pid in df['player_id']:
        # pull match history for this player
        hist = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{pid}/").json()
        df_hist = pd.DataFrame(hist.get("history", []))

        gw = int(gameweek)
        if not df_hist.empty and 'round' in df_hist.columns:
            df_hist['round'] = pd.to_numeric(df_hist['round'], errors='coerce')
            mask = df_hist['round'] == gw
            actual_points = int(df_hist.loc[mask, 'total_points'].sum()) if mask.any() else 0
        else:
            actual_points = 0

        updated.append(actual_points)

    df['actual_points'] = updated
    df.to_csv(csv_path, index=False)
    print(f"Updated actual points for GW{gameweek} in {csv_path}")


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python update_actual_points.py <gameweek>")
        sys.exit(1)

    GW = sys.argv[1]
    update_actual_points(GW)