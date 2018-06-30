import pickle
import numpy as np
import utils


with open('team_power.p', 'rb') as f:
    att_def_power = pickle.load(f)

with open('reg_model_power.p', 'rb') as f:
    reg_1, reg_2 = pickle.load(f)


def make_x(a1, d1, a2, d2, home_advantange):
    return np.array([a1, d1, a2, d2, a1*a2, d1*d2, a1*d2, d1*a2, home_advantange]).reshape(1, -1)

def predict(a1, d1, a2, d2, home_advantage):
    x = make_x(a1, d1, a2, d2, home_advantage)
    return reg_1.predict(x)[0], reg_2.predict(x)[0]


country_1 = 'Panama'
country_2 = 'Tunisia'
home_advantage = 0

a1, d1 = att_def_power[country_1]
a2, d2 = att_def_power[country_2]
e1, e2 = predict(a1, d1, a2, d2, home_advantage)

outcome = utils.outcome_dict(e1, e2)
print(f'{country_1} v {country_2}')
print(f'Expectation outcome: {e1:.3f} - {e2:.3f}')
print(f'Outcome probabilities: {country_1} wins: {outcome["Team 1"]:.5f}, {country_2} wins: {outcome["Team 2"]:.5f}, draw: {outcome["Draw"]:.5f}')
score_outcome_probs = utils.get_score_outcomes(e1, e2)
for score_outcome in score_outcome_probs[:20]:
    print(f'{score_outcome[0]}-{score_outcome[1]}, p={score_outcome[2]:.3f}')
