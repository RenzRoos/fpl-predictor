minutes = {'n_estimators': 1322, 'max_depth': 12, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': True}
goals_scored = {'n_estimators': 827, 'max_depth': 17, 'min_samples_split': 21, 'min_samples_leaf': 12, 'max_features': 'sqrt', 'bootstrap': True}
assists = {'n_estimators': 604, 'max_depth': 24, 'min_samples_split': 5, 'min_samples_leaf': 15, 'max_features': 'sqrt', 'bootstrap': True}
yellow_cards = {'n_estimators': 828, 'max_depth': 28, 'min_samples_split': 12, 'min_samples_leaf': 24, 'max_features': None, 'bootstrap': False}
bonus = {'n_estimators': 1195, 'max_depth': 17, 'min_samples_split': 36, 'min_samples_leaf': 9, 'max_features': 'log2', 'bootstrap': True}
saves = {'n_estimators': 309, 'max_depth': 35, 'min_samples_split': 14, 'min_samples_leaf': 2, 'max_features': None, 'bootstrap': False}

def get_regressor_params(target: str) -> dict:
    if target == "minutes":
        return minutes
    elif target == "goals_scored":
        return goals_scored
    elif target == "assists":
        return assists
    elif target == "yellow_cards":
        return yellow_cards
    elif target == "bonus":
        return bonus
    elif target == "saves":
        return saves
    else:
        raise ValueError(f"No regressor params found for target: {target}")