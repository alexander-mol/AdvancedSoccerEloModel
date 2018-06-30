import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression


import utils

with open('data/2010-2018_patched_df.p', 'rb') as f:
    df = pickle.load(f)

with open('reg_model_simple.p', 'rb') as f:
    reg_1, reg_2 = pickle.load(f)

# with open('standardized_power.p', 'rb') as f:
#     att_def_power = pickle.load(f)
# # convert to strings
# for key in att_def_power:
#     att_def_power[key] = (np.random.random(2))

att_def_power = {}
countries = set(list(df.Country_1.unique()) + list(df.Country_2.unique()))
for country in countries:
    c1_df = df[df.Country_1 == country]
    c2_df = df[df.Country_2 == country]
    n = c1_df.shape[0] + c2_df.shape[0]
    att_def_power[country] = [(c1_df.Score_1.sum()+c2_df.Score_2.sum())/n, (c1_df.Score_2.sum()+c2_df.Score_1.sum())/n]


df['Home_Advantage'] = df.apply(lambda x: 1 if x['Country_1'] in x['Location'] else -1 if x['Country_2'] in x['Location'] else 0, axis=1)

# features = ['A1', 'D1', 'A2', 'D2', 'A1xA2', 'D1xD2', 'A1xD2', 'D1xA2', 'Home_Advantage']
features = ['A1', 'D1', 'A2', 'D2', 'Home_Advantage']

def make_x(a1, d1, a2, d2, home_advantage):
    return np.array([a1, d1, a2, d2, a1 * a2, d1 * d2, a1 * d2, d1 * a2, home_advantage]).reshape(1, -1)

def predict(a1, d1, a2, d2, home_advantage):
    # x = make_x(a1, d1, a2, d2, home_advantage)
    x = np.array([a1, d1, a2, d2, home_advantage]).reshape(1, -1)
    return reg_1.predict(x)[0], reg_2.predict(x)[0]

ADJUSTMENT_RATE = 0.05  # 0.021  # optimized number

# Optimization of adjustment rate
# ADJUSTMENT_RATE = 0.00 gives 20582, 20582 stable after 3 iterations
# ADJUSTMENT_RATE = 0.0005 gives 20562, 20563 decreasing at 0.1 after 100 iterations
# ADJUSTMENT_RATE = 0.001 gives 20568, 20568 decreasing at 0.2 after 60 iterations
# ADJUSTMENT_RATE = 0.002 gives 20579, 20580 decreasing at 0.1 after 48 iterations
# ADJUSTMENT_RATE = 0.003 gives 20593, 20592 decreasing at 0.25 after 30 iterations
# ADJUSTMENT_RATE = 0.01 gives 20670, 20676 pretty stable after 40 iterations


direction_forwards = False
for iter_num in range(1000):  # must be even

    total_error = 0
    # make sure dataframe is sorted in chronological order
    # create features on the fly
    for col in ['A1', 'D1', 'A2', 'D2']:
        df[col] = 0

    if direction_forwards:
        ordered_rows = range(len(df))
    else:
        ordered_rows = reversed(range(len(df)))

    for i in ordered_rows:
        row = df.iloc[i]
        a1, d1 = att_def_power[row.Country_1]
        a2, d2 = att_def_power[row.Country_2]
        df.loc[i, 'A1'], df.loc[i, 'D1'] = a1, d1
        df.loc[i, 'A2'], df.loc[i, 'D2'] = a2, d2
        e1, e2 = predict(a1, d1, a2, d2, row.Home_Advantage)
        total_error += (row.Score_1 - e1) ** 2 + (row.Score_2 - e2) ** 2
        # update team power
        att_def_power[row.Country_1][0] += (row.Score_1 - e1) * ADJUSTMENT_RATE
        att_def_power[row.Country_1][1] += (row.Score_2 - e2) * ADJUSTMENT_RATE

        att_def_power[row.Country_2][0] += (row.Score_2 - e2) * ADJUSTMENT_RATE
        att_def_power[row.Country_2][1] += (row.Score_1 - e1) * ADJUSTMENT_RATE

    print(f'Iteration: {iter_num}, Direction {"forwards" if direction_forwards else "backwards"}, BE score: {att_def_power["Belgium"]}')
    print(total_error)

    # df['A1xA2'] = df['A1'] * df['A2']
    # df['D1xD2'] = df['D1'] * df['D2']
    # df['A1xD2'] = df['A1'] * df['D2']
    # df['D1xA2'] = df['D1'] * df['A2']

    X = df[features]
    y_1 = df.Score_1
    y_2 = df.Score_2
    reg_1 = LinearRegression()
    reg_1.fit(X, y_1)
    print(reg_1.score(X, y_1))
    reg_2 = LinearRegression()
    reg_2.fit(X, y_2)
    print(reg_2.score(X, y_2))

    for i, feature in enumerate(features):
        print(feature, reg_1.coef_[i], reg_2.coef_[i])
    print()
    direction_forwards = not direction_forwards


with open('team_power.p', 'wb') as f:
    pickle.dump(att_def_power, f)

with open('reg_model_power.p', 'wb') as f:
    pickle.dump((reg_1, reg_2), f)
