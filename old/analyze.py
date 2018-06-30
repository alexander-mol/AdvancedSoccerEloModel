import numpy as np
import pickle
from scipy.stats import poisson

score_scaling = 757.5
home_advantage_score = 100.8
expectation_goals = 1.927


def inverse_poisson(probability, mean):
    k0 = 0
    while poisson.cdf(k0+1, mean) < probability:
        k0 += 1
    return (probability - poisson.cdf(k0, mean)) / (poisson.cdf(k0+1, mean) - poisson.cdf(k0, mean)) + k0


def elo_probability(score_diff, score_scaling):
    return 1 / (1 + 10 ** (score_diff / score_scaling))


def get_expectation_scores(elo_1, elo_2, home_advantage=True):
    diff = elo_2 - elo_1 - home_advantage_score * home_advantage
    prob = elo_probability(diff, score_scaling)
    return inverse_poisson(prob, expectation_goals), inverse_poisson(1-prob, expectation_goals)


def outcome_probabilities(elo_1, elo_2, home_advantage=True):
    e1, e2 = get_expectation_scores(elo_1, elo_2, home_advantage)
    pw, pd, pl = 0, 0, 0
    for i in range(20):
        for j in range(20):
            pm = poisson.pmf(i, e1) * poisson.pmf(j, e2)
            if i > j:
                pw += pm
            elif i == j:
                pd += pm
            else:
                pl += pm
    return pw, pd, pl


print(outcome_probabilities(1928, 1684, False))


with open('2010-2018_match_df.p', 'rb') as f:
    df = pickle.load(f)

df['Elo_Score_Before_1'] = df['Elo_Score_New_1'] - df['Elo_Score_Change_1']
df['Elo_Score_Before_2'] = df['Elo_Score_New_2'] - df['Elo_Score_Change_2']

def get_error(params):
    score_scaling = params[0]
    home_advantage = params[1]
    mean_points = params[2]

    df['Elo_Score_Diff'] = df['Elo_Score_Before_2'] - df['Elo_Score_Before_1']
    df['Elo_Score_Diff'] += df.apply(lambda x: -home_advantage * (x['Country_1'] in x['Location']), axis=1)

    df['Elo_Probability'] = df['Elo_Score_Diff'].apply(lambda x: elo_probability(x, score_scaling))
    df['Predicted_Score_1'] = df['Elo_Probability'].apply(lambda x: inverse_poisson(x, mean_points))
    df['Predicted_Score_2'] = df['Elo_Probability'].apply(lambda x: inverse_poisson(1-x, mean_points))

    df['Error'] = (df['Score_1'] - df['Predicted_Score_1']) ** 2 + (df['Score_2'] - df['Predicted_Score_2']) ** 2

    return df['Error'].sum()

# get_error(np.array([400, 100, 1.5]))

# print(minimize(get_error, np.array([400, 100, 1.5])))
optimal_params = np.array([757.5, 100.8, 1.927])
def optimization():
    def derivatives(error_func, params, index, sensitivity=0.001):
        y1 = error_func(params)
        params[index] += sensitivity
        y2 = error_func(params)
        params[index] += sensitivity
        y3 = error_func(params)
        d1 = (y2 - y1) / sensitivity
        d2 = (y3 - y2) / sensitivity
        return (d2 - d1) / sensitivity, (d1 + d2) / 2, y2


    params = optimal_params
    # params = np.array([400, 100, 1.63])
    count = 0
    new_error = None
    while True:
        for i in range(len(params)):
            d__, d_, error = derivatives(get_error, params, i)
            step = - d_ / d__
            params[i] += step
            count += 1
            print(f'{count}: params: {params} error: {error}')

