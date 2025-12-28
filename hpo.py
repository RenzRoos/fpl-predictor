import numpy as np
import pandas as pd
import optuna
import sys
import json
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from utils.request_data import request_data
from utils.get_player_data import get_player_match_history
from utils.config import FEATURES, PLAYER_TARGETS
from sklearn.inspection import permutation_importance


# ----------------------------
# 1) Prevent leakage by forcing "pre-match-safe" features only
# ----------------------------
# Anything that is produced by (or strongly entangled with) what happened in the match
# should NOT be used as a raw per-match feature in HPO.
LEAKY_FEATURES = {
    # direct match outcomes / points components
    "total_points", "bonus", "bps",
    "minutes", "goals_scored", "assists", "yellow_cards", "red_cards",
    "clean_sheets", "goals_conceded", "own_goals",
    "penalties_saved", "penalties_missed", "saves",

    # match-by-match “performance” outputs (still post-match)
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded",

    # match context that is only known after the match
    "team_h_score", "team_a_score",

    # labels derived from whether the player played (outcome)
    "status_played",
}

# The pre-match safe subset you can always know at deadline (given your pipeline):
# - fixture context: was_home, fdr_score
# - availability: chance/status_flag
ALWAYS_SAFE_FEATURES = {
    "was_home",
    "fdr_score",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "status_flag",
}


def permutation_importance_walkforward(
    df: pd.DataFrame,
    FEATURES: list,
    TARGET: str,
    rf_params: dict,
    min_train_rounds: int = 5,
    n_repeats: int = 5,
) -> pd.Series:
    rounds = sorted(df["round"].unique())
    if len(rounds) <= min_train_rounds:
        raise ValueError("Not enough rounds for permutation importance")

    importances = []

    for r in rounds[min_train_rounds:]:
        train = df[df["round"] < r]
        val = df[df["round"] == r]
        if train.empty or val.empty:
            continue

        X_train, y_train = train[FEATURES], train[TARGET]
        X_val, y_val = val[FEATURES], val[TARGET]

        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)

        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=42,
            scoring="neg_mean_absolute_error",
        )

        importances.append(pd.Series(result.importances_mean, index=FEATURES))

    if not importances:
        raise ValueError("Permutation importance failed")

    return pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)


def get_hpo_features(FEATURES: list, TARGET: str) -> list:
    """
    Return the feature list used for HPO after removing leaky columns.
    We keep only features that are pre-match safe.
    """
    # Start from your configured FEATURES, but drop anything leaky or the current TARGET
    filtered = [f for f in FEATURES if f not in LEAKY_FEATURES and f != TARGET]

    # Ensure the always-safe ones are included if present in your data
    # (keeps behavior stable even if FEATURES list changes)
    for f in ALWAYS_SAFE_FEATURES:
        if f not in filtered:
            filtered.append(f)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for f in filtered:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


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

    # ensure columns exist
    for c in FEATURES + [TARGET]:
        if c not in df.columns:
            df[c] = np.nan

    # numeric coercion
    df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["round"] = df["round"].astype(int)

    return df


def walkforward_mae(df: pd.DataFrame, FEATURES: list, TARGET: str, rf_params: dict, min_train_rounds: int = 5) -> float:
    """
    Walk-forward CV on rounds inside df (already < gw).
    Train on rounds < r, validate on round == r.
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

        X_train, y_train = train[FEATURES], train[TARGET]
        X_val, y_val = val[FEATURES], val[TARGET]

        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        maes.append(mean_absolute_error(y_val, pred))

    return float(np.mean(maes)) if maes else float("inf")


def run_hpo_rf_for_gw(data: dict, players_df: pd.DataFrame, gw: int, FEATURES: list, TARGET: str, n_trials: int = 100):
    # leakage-safe features for HPO
    HPO_FEATURES = get_hpo_features(FEATURES, TARGET)

    # Build once with ALL candidate features (so we don't rebuild df every trial)
    df = build_training_table(data, players_df, gw, HPO_FEATURES, TARGET)
    if df.empty:
        raise ValueError(f"No training data found for rounds < GW{gw}")

    def objective(trial: optuna.Trial) -> float:
        # --- NEW: feature subset selection ---
        selected = []
        for f in HPO_FEATURES:
            use_f = trial.suggest_categorical(f"use__{f}", [0, 1])
            if use_f == 1:
                selected.append(f)

        # must have at least 1 feature
        if len(selected) == 0:
            return float("inf")

        # --- existing RF hyperparams ---
        rf_params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            max_depth=trial.suggest_int("max_depth", 3, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 50),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 25),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            random_state=42,
            n_jobs=-1,
        )

        return walkforward_mae(df, selected, TARGET, rf_params, min_train_rounds=5)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # --- NEW: recover best feature subset from best trial params ---
    best_trial_params = study.best_trial.params
    best_features = [f for f in HPO_FEATURES if best_trial_params.get(f"use__{f}", 0) == 1]

    # remove the use__* flags from rf params for saving
    best_rf_params = {k: v for k, v in best_trial_params.items() if not k.startswith("use__")}

    return {
        "gw_cutoff": int(gw),
        "target": TARGET,
        "hpo_features_candidates": HPO_FEATURES,
        "best_features": best_features,
        "best_mae": float(study.best_value),
        "best_rf_params": best_rf_params,
    }



if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python hpo_setup.py <GW>")
    gw = int(sys.argv[1])

    data = request_data("https://fantasy.premierleague.com/api/bootstrap-static/")
    players = pd.DataFrame(data["elements"]).copy()

    # simpler and safer than .where(...).dropna(...)
    players = players[pd.to_numeric(players["minutes"], errors="coerce").fillna(0) > 300].copy()
    players["id"] = players["id"].astype(int)

    os.makedirs("data/hpo_results", exist_ok=True)

    for target in PLAYER_TARGETS:
        res = run_hpo_rf_for_gw(
            data, players, gw, FEATURES, TARGET=target, n_trials=200
        )

        df_train = build_training_table(data, players, gw, FEATURES, target)

        perm_imp = permutation_importance_walkforward(
            df_train,
            FEATURES,
            TARGET=target,
            rf_params=res["best_rf_params"],
        )

        # save results
        with open(f"data/hpo_results/hpo_gw{gw}_{target}.txt", "w") as f:
            f.write(f"best_mae: {res['best_mae']:.4f}\n")
            f.write(f"best_rf_params: {res['best_rf_params']}\n\n")
            f.write("Permutation importance (MAE increase):\n")
            for feat, val in perm_imp.items():
                f.write(f"{feat}: {val:.6f}\n")

        print(f"[{target}] best_mae={res['best_mae']:.4f}")