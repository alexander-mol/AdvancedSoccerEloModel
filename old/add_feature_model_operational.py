import numpy as np
import pickle
import utils

# MATCH DETAILS
country_1 = 'England'
country_2 = 'Panama'
home_advantage = False
friendly = False

elo_1 = utils.get_elo(country_1)
elo_2 = utils.get_elo(country_2)

with open('average_goals_dict.p', 'rb') as f:
    average_goals_dict = pickle.load(f)

# create_features
"""
features = ['Elo_Score_Before_1', 'Elo_Score_Before_2', 'Elo_Score_Diff', 'Score_Diff^2', 'LN(Elo_Score_Before_1)',
            'LN(Elo_Score_Before_2)', 'Elo_Score_Product', 'Friendly_Flag', 'Home_Advantage',
            'Home_AdvantagexElo_Score_Diff',
            'Avg_For_1', 'Avg_Against_1', 'Avg_For_2', 'Avg_Against_2', 'Avg_For_1xAvg_For_2',
            'Avg_Against_1xAvg_Against_2', 'Avg_For_1xAvg_Against_2', 'Avg_Against_1xAvg_For_2']
"""

def make_x(elo_1, elo_2, country_1, country_2):
    return np.array([
        elo_1,
        elo_2,
        elo_2 - elo_1,
        (elo_2 - elo_1) ** 2,
        np.log(elo_1),
        np.log(elo_2),
        elo_1 * elo_2,
        friendly*1,
        home_advantage*1,
        home_advantage*(elo_2 - elo_1),
        average_goals_dict[country_1][0],
        average_goals_dict[country_1][1],
        average_goals_dict[country_2][0],
        average_goals_dict[country_2][1],
        average_goals_dict[country_1][0] * average_goals_dict[country_2][0],
        average_goals_dict[country_1][1] * average_goals_dict[country_2][1],
        average_goals_dict[country_1][0] * average_goals_dict[country_2][1],
        average_goals_dict[country_1][1] * average_goals_dict[country_2][0],
    ]).reshape(1, -1)

with open('reg_model_add_features.p', 'rb') as f:
    reg_1, reg_2 = pickle.load(f)

x1, x2 = make_x(elo_1, elo_2, country_1, country_2), make_x(elo_2, elo_1, country_2, country_1)
e1, e2 = (reg_1.predict(x1)[0] + reg_2.predict(x2)[0]) / 2, (reg_2.predict(x1)[0] + reg_1.predict(x2)[0]) / 2
outcome = utils.outcome_dict(e1, e2)

print(f'{country_1} v {country_2} - Elo {elo_1} v {elo_2}, avg result ({average_goals_dict[country_1][0]:.2f}, {average_goals_dict[country_1][1]:.2f}) v ({average_goals_dict[country_2][0]:.2f}, {average_goals_dict[country_2][1]:.2f})')
print(f'Expectation outcome: {e1:.3f} - {e2:.3f}')
print(f'Outcome probabilities: {country_1} wins: {outcome["Team 1"]:.5f}, {country_2} wins: {outcome["Team 2"]:.5f}, draw: {outcome["Draw"]:.5f}')
score_outcome_probs = utils.get_score_outcomes(e1, e2)
for score_outcome in score_outcome_probs[:20]:
    print(f'{score_outcome[0]}-{score_outcome[1]}, p={score_outcome[2]:.3f}')
