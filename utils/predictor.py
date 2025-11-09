import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests

from utils.get_player_data import get_player_match_history
from utils.feature_rows import build_X_pred_for_gw
from utils.play_probability import get_play_probability

def predict_gameweek(data: pd.DataFrame, players_df: pd.DataFrame, gw: int, FEATURES: list, TARGET: str, N_RUNS: int) -> pd.DataFrame:
    results = []

    for pid, name in zip(players_df['id'], players_df['web_name']):
        mh = get_player_match_history(data, pid)
        if mh is None or mh.empty or 'round' not in mh.columns:
            continue

        mh['round'] = pd.to_numeric(mh['round'], errors='coerce')
        mh = mh.dropna(subset=['round'])

        train_df = mh[mh['round'] != gw].copy()
        if len(train_df) < 2:
            continue

        train_df['status_played'] = (train_df['status'] == 'Played').astype(int)
        train_df['was_home'] = train_df['was_home'].astype(int)

        for c in FEATURES + [TARGET]:
            if c not in train_df.columns:
                train_df[c] = np.nan

        train_df[FEATURES + [TARGET]] = train_df[FEATURES + [TARGET]].apply(pd.to_numeric, errors='coerce').fillna(0)

        X = train_df[FEATURES]
        y = train_df[TARGET]

        if len(X) >= 5:
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, y_train = X, y

        X_pred = build_X_pred_for_gw(mh, players_df, pid, gw, FEATURES, TARGET)
        if X_pred is None:
            continue

        preds = []
        for _ in range(N_RUNS):
            model = RandomForestRegressor(n_estimators=200, random_state=random.randint(1, 10_000))
            model.fit(X_train, y_train)
            preds.append(float(model.predict(X_pred)[0]))
        mean_pred = float(np.mean(preds))

        bootstrap = data
        player_meta = pd.DataFrame(bootstrap["elements"])[
            ["id","status","chance_of_playing_next_round","chance_of_playing_this_round"]
        ]           
        
        meta_row = player_meta.loc[player_meta["id"] == pid].iloc[0] if (player_meta["id"] == pid).any() else None
        p_play = get_play_probability(meta_row, gw)
        if np.isnan(p_play):
            p_play = 100.0

        mean_pred *= (p_play / 100.0)

        results.append({
            "player_id": int(pid),
            "player_name": name,
            "round": int(gw),
            "predicted_points": mean_pred
        })

    df_out = pd.DataFrame(results)
    if df_out.empty or 'predicted_points' not in df_out.columns:
        return df_out.reset_index(drop=True)
    return df_out.sort_values("predicted_points", ascending=False).reset_index(drop=True)
