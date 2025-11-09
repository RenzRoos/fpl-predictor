import pandas as pd
import sys

def create_team(GW: int, predictions_df: pd.DataFrame, players: pd.DataFrame):
    # Prepare data with position + team + availability
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    pred_pos = predictions_df.merge(
        players[["id", "element_type", "team", "status", "chance_of_playing_next_round"]],
        left_on="player_id", right_on="id", how="left"
    ).drop(columns=["id"])
    pred_pos["position"] = pred_pos["element_type"].map(pos_map)

    # Availability filter
    pred_pos = pred_pos[
        (pred_pos["chance_of_playing_next_round"].fillna(100) >= 50)
        & (~pred_pos["status"].isin(["i", "s", "u"]))
    ].copy()

    # Keep only needed columns
    pred_pos["predicted_points"] = pd.to_numeric(pred_pos["predicted_points"], errors="coerce").fillna(0.0)
    pred_pos = pred_pos[["player_id","player_name","position","round","predicted_points","team"]].reset_index(drop=True)

    # Required counts per position
    req = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    teams = pred_pos["team"].unique().tolist()

    # --- Try exact optimization with ILP ---
    try:
        import pulp

        # Decision vars
        idx = list(pred_pos.index)
        x = pulp.LpVariable.dicts("pick", idx, lowBound=0, upBound=1, cat="Binary")

        prob = pulp.LpProblem("FPL_Squad_Select", pulp.LpMaximize)
        # Objective
        prob += pulp.lpSum(pred_pos.loc[i, "predicted_points"] * x[i] for i in idx)

        # Position constraints
        for p, k in req.items():
            prob += pulp.lpSum(x[i] for i in idx if pred_pos.loc[i, "position"] == p) == k

        # Team cap ≤ 3
        for t in teams:
            prob += pulp.lpSum(x[i] for i in idx if pred_pos.loc[i, "team"] == t) <= 3

        # Solve
        _ = prob.solve(pulp.PULP_CBC_CMD(msg=False))

        chosen = [i for i in idx if pulp.value(x[i]) == 1]
        squad_df = pred_pos.loc[chosen, ["player_id","player_name","position","round","predicted_points"]]
        squad_df = squad_df.sort_values(["position","predicted_points"], ascending=[True, False]).reset_index(drop=True)

    except Exception:
        # --- Greedy fallback (feasible, not guaranteed optimal) ---
        remaining = req.copy()
        team_count = {t: 0 for t in teams}
        chosen_rows = []

        for _, row in pred_pos.sort_values("predicted_points", ascending=False).iterrows():
            pos = row["position"]; team = row["team"]
            if remaining.get(pos, 0) > 0 and team_count[team] < 3:
                chosen_rows.append(row)
                remaining[pos] -= 1
                team_count[team] += 1
                if sum(remaining.values()) == 0:
                    break

        squad_df = pd.DataFrame(chosen_rows)[["player_id","player_name","position","round","predicted_points"]]
        squad_df = squad_df.sort_values(["position","predicted_points"], ascending=[True, False]).reset_index(drop=True)

    # Save
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