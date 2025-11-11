import requests
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def get_avg_manager_score_single(gw: int) -> int:
    data = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    events = pd.DataFrame(data["events"])
    row = events.loc[events["id"] == gw, "average_entry_score"]
    return int(row.iloc[0]) if not row.empty else 0

def get_scout_picks_score(gw: int) -> int:
    squad_path = f"scout_picks/gw{gw}_scout_picks.csv"
    df = pd.read_csv(squad_path)
    points = df["actual_points"].sum()
    return int(points)

def get_predicted_squad_score(gw: int) -> int:
    squad_path = f"teams/gw{gw}_squad.csv"
    df = pd.read_csv(squad_path)
    points = df[df["is_starter"] == 1]["actual_points"].sum()
    return int(points)

def compare_scores(gw: int, all: bool):
    if all:
        avg_avg_scores, avg_scout_scores, avg_predicted_scores = 0, 0, 0
        for gw in range(1, len(os.listdir("teams")) + 1):
            avg_avg_scores += get_avg_manager_score_single(gw)
            avg_scout_scores += get_scout_picks_score(gw)
            avg_predicted_scores += get_predicted_squad_score(gw)
        total_gws = len(os.listdir("teams"))
        print(f"Average Manager Score across all GWs: {avg_avg_scores / total_gws}")
        print(f"Scout Picks Actual Score across all GWs: {avg_scout_scores / total_gws}")   
        print(f"Predicted Squad Score across all GWs: {avg_predicted_scores / total_gws}")

        width = 0.25
        x = np.arange(1, total_gws + 1)

        avg_scores = [get_avg_manager_score_single(gw) for gw in x]
        scout_scores = [get_scout_picks_score(gw) for gw in x]
        predicted_scores = [get_predicted_squad_score(gw) for gw in x]

        plt.bar(x - width, avg_scores, width=width, alpha=0.6, label="Average Manager Score")
        plt.bar(x, scout_scores, width=width, alpha=0.6, label="Scout Picks Actual Score")
        plt.bar(x + width, predicted_scores, width=width, alpha=0.6, label="Predicted Squad Score")

        plt.xlabel("Gameweek")
        plt.ylabel("Score")
        plt.title("Score Comparison per Gameweek")
        plt.xticks(x)
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        avg_score = get_avg_manager_score_single(gw)
        scout_score = get_scout_picks_score(gw)
        predicted_score = get_predicted_squad_score(gw)
        print(f"Gameweek {gw} Comparison:")
        print(f"Average Manager Score: {avg_score}")
        print(f"Scout Picks Actual Score: {scout_score}")   
        print(f"Predicted Squad Score: {predicted_score}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare.py <gameweek> | -a")
        sys.exit(1)

    if not sys.argv[1].isdigit() and sys.argv[1] != "-a":
        print("Gameweek must be an integer.")
        sys.exit(1)

    gw = int(sys.argv[1])
    if sys.argv[1] == "-a":
        compare_scores(gw, all=True)
    else:
        compare_scores(gw, all=False)
