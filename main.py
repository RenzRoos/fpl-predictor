import sys
import pandas as pd
import requests

from utils.predictor import predict_gameweek
from utils.select_team import create_team

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python main.py <GW>")

    GW = int(sys.argv[1])

    data = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    players = pd.DataFrame(data['elements'])

    FEATURES = [
        'was_home','status_played','minutes','fdr_score','goals_scored','assists',
        'clean_sheets','goals_conceded','own_goals','penalties_saved','penalties_missed',
        'yellow_cards','bps','influence','creativity','red_cards','saves','ict_index',
        'expected_goals','expected_assists','expected_goal_involvements','expected_goals_conceded',
        'chance_of_playing_next_round', 'chance_of_playing_this_round', 'status_flag'
    ]
    TARGET = 'total_points'

    predictions_df = predict_gameweek(data, players, GW, FEATURES, TARGET, N_RUNS=5)

    out_path = f"data/gw{GW}_predicted_points.csv"
    predictions_df.to_csv(out_path, columns=["player_id","player_name","round","predicted_points"], index=False)
    print(f"Saved predictions to {out_path}")

    create_team(GW, predictions_df, players)

if __name__ == "__main__":
    main()
