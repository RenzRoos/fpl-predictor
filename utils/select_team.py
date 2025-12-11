import pandas as pd
import sys
import pandas as pd

def create_team(GW: int, predictions_df: pd.DataFrame, players: pd.DataFrame):

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

    # --- SQUAD: top-N per position (no team cap) ---
    gks  = df.loc[df["position"] == "GK"].nlargest(2, "predicted_points")
    defs = df.loc[df["position"] == "DEF"].nlargest(5, "predicted_points")
    mids = df.loc[df["position"] == "MID"].nlargest(5, "predicted_points")
    fwds = df.loc[df["position"] == "FWD"].nlargest(3, "predicted_points")

    squad_df = pd.concat([gks, defs, mids, fwds], ignore_index=True).copy()

    # --- STARTING XI: 1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD, then best remaining ---
    xi_idx = []

    # required minimums
    xi_idx += list(squad_df[squad_df["position"]=="GK"].nlargest(1, "predicted_points").index)
    xi_idx += list(squad_df.drop(index=xi_idx).loc[squad_df["position"]=="DEF"].nlargest(3, "predicted_points").index)
    xi_idx += list(squad_df.drop(index=xi_idx).loc[squad_df["position"]=="MID"].nlargest(2, "predicted_points").index)
    xi_idx += list(squad_df.drop(index=xi_idx).loc[squad_df["position"]=="FWD"].nlargest(1, "predicted_points").index)

    # fill remaining slots to 11 with highest predicted points
    need = 11 - len(xi_idx)
    if need > 0:
        xi_idx += list(squad_df.drop(index=xi_idx).nlargest(need, "predicted_points").index)

    squad_df["is_starter"] = squad_df.index.isin(xi_idx).astype(int)

    # (optional) bench ordering: GK first, then by predicted points
    bench = squad_df[squad_df["is_starter"]==0].copy()
    bench_gk  = bench[bench["position"]=="GK"]
    bench_out = bench[bench["position"]!="GK"].sort_values("predicted_points", ascending=False)
    bench = pd.concat([bench_gk, bench_out], ignore_index=True)
    bench["bench_order"] = range(1, len(bench)+1)

    starters_df = squad_df[squad_df["is_starter"]==1].copy()
    starters_df["bench_order"] = 0
    final = pd.concat([starters_df, bench], ignore_index=True)

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