# FEATURES = [
#     'was_home','status_played','minutes','fdr_score','goals_scored','assists',
#     'clean_sheets','goals_conceded','own_goals','penalties_saved','penalties_missed',
#     'yellow_cards','bps','influence','creativity','red_cards','saves','ict_index',
#     'expected_goals','expected_assists','expected_goal_involvements','expected_goals_conceded',
#     'chance_of_playing_next_round', 'chance_of_playing_this_round', 'status_flag'
# ]
# TARGET = 'total_points'

FEATURES = [
    'was_home',
    'fdr_score',
    'status_played',
    'chance_of_playing_next_round',
    'chance_of_playing_this_round',
    'status_flag',
    'ict_index',
    'influence',
    'creativity',
    'threat',
    'expected_goals',
    'expected_assists',
    'expected_goal_involvements'
]

PLAYER_TARGETS = [
    'minutes',
    'goals_scored',
    'assists',
    'yellow_cards',
    'bps',
    'saves',
]

TEAM_TARGET = 'goals_conceded'
