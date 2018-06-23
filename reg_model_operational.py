import numpy as np
from scipy.stats import poisson

def expectation_scores(elo1, elo2, home_advantage_flag=False, friendly_flag=False):
    # ['Elo_Score_Before_1', 'Elo_Score_Before_2', 'Elo_Score_Diff', 'Score_Diff^2', 'LN(Elo_Score_Before_1)',
     # 'LN(Elo_Score_Before_2)', 'Elo_Score_Product', 'LN(Elo_Score_Product)', 'Friendly_Flag', 'Home_Advantage']
    reg_1_coef = np.array([ -1.51706890e-04,  -7.14916567e-04,  -5.63209014e-04,
         2.04056684e-06,   1.95210952e+00,  -2.32556890e+00,
         4.92255602e-07,  -3.73459376e-01,  -3.78653821e-02,
         2.42091156e-01, 9.5087171838605755])
    reg_2_coef = np.array([ -2.62419413e-04,   1.05159679e-04,   3.67578303e-04,
         2.05915042e-06,  -2.81064538e+00,   2.13543666e+00,
         5.22194299e-07,  -6.75208715e-01,  -2.89518161e-02,
        -2.56216351e-01, 14.977599474304611])
    features = np.array(
        [elo1, elo2, elo2 - elo1, (elo2 - elo1) ** 2, np.log(elo1), np.log(elo2), elo1 * elo2, np.log(elo1*elo2), friendly_flag * 1,
         home_advantage_flag * 1, 1])
    e1 = np.dot(reg_1_coef, features)
    e2 = np.dot(reg_2_coef, features)
    return e1, e2


def outcome_probabilities(e1, e2):
    pw, pd, pl = 0, 0, 0
    for i in range(30):
        for j in range(30):
            pm = poisson.pmf(i, e1) * poisson.pmf(j, e2)
            if i > j:
                pw += pm
            elif i == j:
                pd += pm
            else:
                pl += pm
    return {'Team 1 wins': pw, 'Team 2 wins': pl, 'Draw': pd}


def predict(elo1, elo2, home_advantage=True, friendly_flag=False):
    e1_, e2_ = expectation_scores(elo1, elo2, home_advantage, friendly_flag)
    e1__, e2__ = expectation_scores(elo2, elo1, home_advantage, friendly_flag)
    e1, e2 = (e1_ + e2__) / 2, (e2_ + e1__) / 2
    output = f'Expectation scores: {e1:.3f} - {e2:.3f}. \nOutcome probabilities: \n'
    outcome_probs = outcome_probabilities(e1, e2)
    for key in outcome_probs:
        output += f'{key}: {outcome_probs[key]:.5f}\n'
    print(output)
    return output

predict(1956, 1648, True)
