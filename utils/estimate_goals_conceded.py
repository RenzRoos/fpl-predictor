import pandas as pd

from utils.request_data import request_data

def estimate_team_goals_conceded(data: dict, gw: int) -> pd.Series:
    """
    Return a Series indexed by team_id with expected goals conceded for GW `gw`,
    using only:
      - whether the team is home/away
      - opponent attack strength
      - historical goals conceded per game
    """
    teams_meta = pd.DataFrame(data["teams"])[[
        "id", "strength_attack_home", "strength_attack_away"
    ]].set_index("id")

    # all fixtures with results, before this GW
    all_fx = pd.DataFrame(
        request_data("https://fantasy.premierleague.com/api/fixtures/")
    )
    hist = all_fx[
        (all_fx["finished"] == True)
        & all_fx["event"].notna()
    ].copy()

    if hist.empty:
        # no history yet: fallback to league-average 1.5 goals per game
        return pd.Series(1.5, index=teams_meta.index, name="predicted_goals_conceded")

    # compute historical goals conceded per team
    hist["team_h_gc"] = hist["team_a_score"]
    hist["team_a_gc"] = hist["team_h_score"]

    home_gc = hist[["team_h", "team_h_gc"]].rename(
        columns={"team_h": "team_id", "team_h_gc": "gc"}
    )
    away_gc = hist[["team_a", "team_a_gc"]].rename(
        columns={"team_a": "team_id", "team_a_gc": "gc"}
    )
    gc_all = pd.concat([home_gc, away_gc], ignore_index=True)

    gc_per_team = gc_all.groupby("team_id")["gc"].mean()
    league_gc_mean = gc_all["gc"].mean()

    # average opponent attack strength in league (for normalization)
    avg_att = float(
        (teams_meta["strength_attack_home"].mean()
         + teams_meta["strength_attack_away"].mean()) / 2.0
    )

    # fixtures of this GW only (can be future)
    gw_fx = pd.DataFrame(
        request_data(f"https://fantasy.premierleague.com/api/fixtures/?event={gw}")
    )
    if gw_fx.empty:
        # nothing scheduled? just return historical mean
        return gc_per_team.reindex(teams_meta.index).fillna(league_gc_mean)

    team_gc_pred = {}

    for _, row in gw_fx.iterrows():
        th = int(row["team_h"])
        ta = int(row["team_a"])

        # base rates from history (fallback → league mean)
        base_h = gc_per_team.get(th, league_gc_mean)
        base_a = gc_per_team.get(ta, league_gc_mean)

        # opponent attack strengths (venue-aware)
        opp_att_vs_home = teams_meta.loc[ta, "strength_attack_away"]
        opp_att_vs_away = teams_meta.loc[th, "strength_attack_home"]

        # simple scaling: λ = base_gc * (opp_att / avg_att)
        lam_h = base_h * (opp_att_vs_home / avg_att)
        lam_a = base_a * (opp_att_vs_away / avg_att)

        team_gc_pred[th] = float(lam_h)
        team_gc_pred[ta] = float(lam_a)

    # make sure all teams have some value
    s = pd.Series(team_gc_pred, name="predicted_goals_conceded")
    s = s.reindex(teams_meta.index).fillna(league_gc_mean)
    return s
