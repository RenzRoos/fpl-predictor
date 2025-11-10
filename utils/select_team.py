import pandas as pd
import sys

def create_team(GW: int, predictions_df: pd.DataFrame, players: pd.DataFrame):
    import pandas as pd

    # --- prepare data with position + team + availability ---
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    df = predictions_df.merge(
        players[["id", "element_type", "team", "status", "chance_of_playing_next_round"]],
        left_on="player_id", right_on="id", how="left"
    ).drop(columns=["id"]).copy()
    df["position"] = df["element_type"].map(pos_map)

    # availability filter
    df = df[
        (df["chance_of_playing_next_round"].fillna(100) >= 50)
        & (~df["status"].isin(["i", "s", "u"]))  # injured/suspended/unavailable
    ].copy()

    # keep needed cols and clean
    df["predicted_points"] = pd.to_numeric(df["predicted_points"], errors="coerce").fillna(0.0)
    df = df[["player_id","player_name","position","round","predicted_points","team"]].reset_index(drop=True)

    # required squad counts per position (15 total)
    req = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    teams = df["team"].unique().tolist()
    idx = list(df.index)

    # try exact ILP
    try:
        import pulp

        # decision variables
        y = pulp.LpVariable.dicts("in_squad", idx, lowBound=0, upBound=1, cat="Binary")
        s = pulp.LpVariable.dicts("starter",  idx, lowBound=0, upBound=1, cat="Binary")

        prob = pulp.LpProblem("FPL_Squad_And_Lineup", pulp.LpMaximize)

        # objective: prioritize starters; tiny weight for bench quality
        prob += pulp.lpSum(df.loc[i, "predicted_points"] * s[i] for i in idx)

        # squad position counts (exact)
        for p, k in req.items():
            prob += pulp.lpSum(y[i] for i in idx if df.loc[i, "position"] == p) == k

        # team cap for whole squad
        for t in teams:
            prob += pulp.lpSum(y[i] for i in idx if df.loc[i, "team"] == t) <= 3

        # starters: exactly 11
        prob += pulp.lpSum(s[i] for i in idx) == 11

        # starter formation constraints
        prob += pulp.lpSum(s[i] for i in idx if df.loc[i, "position"] == "GK") == 1
        prob += pulp.lpSum(s[i] for i in idx if df.loc[i, "position"] == "DEF") >= 3
        prob += pulp.lpSum(s[i] for i in idx if df.loc[i, "position"] == "MID") >= 2
        prob += pulp.lpSum(s[i] for i in idx if df.loc[i, "position"] == "FWD") >= 1

        # starters must be in squad
        for i in idx:
            prob += s[i] <= y[i]

        # solve
        _ = prob.solve(pulp.PULP_CBC_CMD(msg=False))

        chosen = [i for i in idx if pulp.value(y[i]) == 1]
        starters = [i for i in idx if pulp.value(s[i]) == 1]

        squad_df = df.loc[chosen].copy()
        squad_df["is_starter"] = squad_df.index.isin(starters).astype(int)

    except Exception:
        # greedy fallback (feasible, not guaranteed optimal)
        remaining = req.copy()
        team_count = {t: 0 for t in teams}
        chosen_rows = []

        for _, row in df.sort_values("predicted_points", ascending=False).iterrows():
            pos = row["position"]; team = row["team"]
            if remaining.get(pos, 0) > 0 and team_count[team] < 3:
                chosen_rows.append(row)
                remaining[pos] -= 1
                team_count[team] += 1
                if sum(remaining.values()) == 0:
                    break

        squad_df = pd.DataFrame(chosen_rows).copy()

        # starters: enforce GK=1, DEF>=3, MID>=2, FWD>=1
        starters = []
        # GK first
        starters += list(
            squad_df.loc[squad_df["position"]=="GK"].nlargest(1,"predicted_points").index
        )
        # DEF at least 3
        starters += list(
            squad_df.drop(index=starters).loc[squad_df["position"]=="DEF"].nlargest(3,"predicted_points").index
        )
        # MID at least 2
        starters += list(
            squad_df.drop(index=starters).loc[squad_df["position"]=="MID"].nlargest(2,"predicted_points").index
        )
        # FWD at least 1
        starters += list(
            squad_df.drop(index=starters).loc[squad_df["position"]=="FWD"].nlargest(1,"predicted_points").index
        )
        # fill up to 11 best remaining
        need = 11 - len(starters)
        if need > 0:
            starters += list(
                squad_df.drop(index=starters).nlargest(need,"predicted_points").index
            )

        squad_df["is_starter"] = squad_df.index.isin(starters).astype(int)

    # bench ordering: highest predicted points first among bench
    bench = squad_df[squad_df["is_starter"]==0].copy().sort_values("predicted_points", ascending=False)
    starters_df = squad_df[squad_df["is_starter"]==1].copy()

    final = pd.concat([starters_df, bench], ignore_index=True)
    final = final.sort_values(
        by=["is_starter","position","predicted_points"],
        ascending=[False, True, False]
    )

    # save
    squad_path = f"teams/gw{GW}_squad.csv"
    cols = ["player_id","player_name","position","round", "is_starter","predicted_points"]
    final[cols].to_csv(squad_path, index=False)
    print(f"Saved squad + lineup to {squad_path}")




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

    print()
    print("-------------------------------")
    print(f"Total predicted points for GW{gameweek} squad: {total_predicted_points:.1f}")
    print(f"Total actual points for GW{gameweek} squad: {total_actual_points:.1f}")
    print(f"Difference: {total_actual_points - total_predicted_points:.1f}")
    print()

    xi = team[team["is_starter"] == 1].copy()

    total_actual_points = xi["actual_points"].sum()
    total_predicted_points = xi["predicted_points"].fillna(0).sum()


    print(f"Total predicted points for GW{gameweek} starting XI: {total_predicted_points:.1f}")
    print(f"Total actual points for GW{gameweek} starting XI: {total_actual_points:.1f}")
    print(f"Difference (XI): {total_actual_points - total_predicted_points:.1f}")
    print("-------------------------------")
    print()

    team.to_csv(squad_path, index=False)
    print(f"Updated {squad_path} with actual_points column.")

    return total_actual_points


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python select_team.py <GW>")

    GW = int(sys.argv[1])

    evaluate_team_performance(GW)