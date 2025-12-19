import numpy as np
import pandas as pd
import optuna
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from utils.get_player_data import get_player_match_history
from utils.request_data import request_data

def build_training_table(data: dict, players_df: pd.DataFrame, gw: int, FEATURES: list, TARGET: str) -> pd.DataFrame:
    """Concatenate all players' match history, keeping only rounds < gw."""
    rows = []
    for pid, name in zip(players_df["id"], players_df["web_name"]):
        mh = get_player_match_history(data, int(pid))
        if mh is None or mh.empty or "round" not in mh.columns:
            continue

        mh = mh.copy()
        mh["round"] = pd.to_numeric(mh["round"], errors="coerce")
        mh = mh.dropna(subset=["round"])
        mh = mh[mh["round"] < gw]  
        if mh.empty:
            continue

        mh["player_id"] = int(pid)
        mh["player_name"] = name
        rows.append(mh)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)

    # ensure cols exist + numeric
    for c in FEATURES + [TARGET]:
        if c not in df.columns:
            df[c] = np.nan
    df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["round"] = df["round"].astype(int)

    return df


def walkforward_mae(df: pd.DataFrame, used_features: list, target: str, rf_params: dict, min_train_rounds: int = 5) -> float:
    """
    Evaluate using walk-forward:
      for each validation round r in [min_train_rounds .. last_round],
      train on rounds < r, validate on round == r.
    """
    rounds = sorted(df["round"].unique())
    if len(rounds) <= min_train_rounds:
        return float("inf")

    maes = []
    for r in rounds[min_train_rounds:]:
        train = df[df["round"] < r]
        val = df[df["round"] == r]
        if train.empty or val.empty:
            continue

        X_train, y_train = train[used_features], train[target]
        X_val, y_val = val[used_features], val[target]

        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, pred))

    return float(np.mean(maes)) if maes else float("inf")


def run_hpo_for_gw(data: dict, players_df: pd.DataFrame, gw: int, FEATURES: list, TARGET: str, n_trials: int = 100):
    df = build_training_table(data, players_df, gw, FEATURES, TARGET)
    if df.empty:
        raise ValueError(f"No training data found for rounds < GW{gw}")

    def objective(trial: optuna.Trial) -> float:
        # --- feature subset search ---
        used = []
        for f in FEATURES:
            if trial.suggest_int(f"use_{f}", 0, 1) == 1:
                used.append(f)

        # guardrails (avoid empty / silly configs)
        if len(used) < 3:
            return float("inf")

        # --- RF hyperparameters ---
        rf_params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            max_depth=trial.suggest_int("max_depth", 3, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 40),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=42,
            n_jobs=-1,
        )

        return walkforward_mae(df, used, TARGET, rf_params, min_train_rounds=5)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_features = [f for f in FEATURES if best_params.get(f"use_{f}", 0) == 1]

    # strip feature flags out of params
    best_rf_params = {k: v for k, v in best_params.items() if not k.startswith("use_")}

    return {
        "best_mae": study.best_value,
        "best_features": best_features,
        "best_rf_params": best_rf_params,
    }

if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python hpo_setup.py <GW>")
    gw = int(sys.argv[1])

    FEATURES = [
        'was_home','status_played','minutes','fdr_score','goals_scored','assists',
        'clean_sheets','goals_conceded','own_goals','penalties_saved','penalties_missed',
        'yellow_cards','bps','influence','creativity','red_cards','saves','ict_index',
        'expected_goals','expected_assists','expected_goal_involvements','expected_goals_conceded',
        'chance_of_playing_next_round','chance_of_playing_this_round','status_flag'
    ]
    TARGET = "total_points"

    data = request_data("https://fantasy.premierleague.com/api/bootstrap-static/")
    players = pd.DataFrame(data['elements'])
    players = players.where(players['minutes'] > 300).dropna(subset=['minutes'])
    players["id"] = players["id"].astype(int)

    result = run_hpo_for_gw(data, players, gw=gw, FEATURES=FEATURES, TARGET=TARGET, n_trials=200)
    print(result["best_mae"])
    print(result["best_features"])
    print(result["best_rf_params"])
