import pandas as pd
import sys

def create_team(GW: int, predictions_df: pd.DataFrame, players: pd.DataFrame):
    print(predictions_df.head())
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    pred_pos = predictions_df.merge(
        players[["id", "element_type", "status", "chance_of_playing_next_round"]],
        left_on="player_id", right_on="id", how="left"
    ).drop(columns=["id"])
    pred_pos["position"] = pred_pos["element_type"].map(pos_map)

    pred_pos = pred_pos[
        (pred_pos["chance_of_playing_next_round"].fillna(100) >= 50)
        & (~pred_pos["status"].isin(["i", "s", "u"]))  # i=injured, s=suspended, u=unavailable
    ]

    gks  = pred_pos.loc[pred_pos["position"] == "GK"].nlargest(2, "predicted_points")
    defs = pred_pos.loc[pred_pos["position"] == "DEF"].nlargest(5, "predicted_points")
    mids = pred_pos.loc[pred_pos["position"] == "MID"].nlargest(5, "predicted_points")
    fwds = pred_pos.loc[pred_pos["position"] == "FWD"].nlargest(3, "predicted_points")

    squad_df = pd.concat([gks, defs, mids, fwds], ignore_index=True)
    squad_df = squad_df[
        ["player_id","player_name","position","round","predicted_points"]
    ].sort_values(
        ["position","predicted_points"], ascending=[True, False]
    ).reset_index(drop=True)

    squad_path = f"teams/gw{GW}_squad.csv"
    squad_df.to_csv(squad_path, index=False)
    print(f"Saved squad to {squad_path}")



def evaluate_team_performance(gameweek: int):
    """Sum actual points of the chosen 15-man squad and update the file."""
    squad_path = f"teams/gw{gameweek}_squad.csv"
    df = pd.read_csv(squad_path)

    pred_path = f"data/gw{gameweek}_predicted_points.csv"
    all_players = pd.read_csv(pred_path)[["player_id", "actual_points"]]

    team = df.merge(all_players, on="player_id", how="left")
    team["actual_points"] = team["actual_points"].fillna(0)

    total_actual_points = team["actual_points"].sum()
    total_predicted_points = team["predicted_points"].fillna(0).sum()

    print(f"Total predicted points for GW{gameweek} squad: {total_predicted_points:.1f}")
    print(f"Total actual points for GW{gameweek} squad: {total_actual_points:.1f}")
    print(f"Difference: {total_actual_points - total_predicted_points:.1f}")

    team.to_csv(squad_path, index=False)
    print(f"Updated {squad_path} with actual_points column.")

    return total_actual_points


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python select_team.py <GW>")

    GW = int(sys.argv[1])

    # Example usage (assuming predictions_df and players DataFrame are available)
    # create_team(GW, predictions_df, players)

    # To evaluate team performance
    evaluate_team_performance(GW)