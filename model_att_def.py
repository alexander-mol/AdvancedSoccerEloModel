import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression


import utils

with open('2010-2018_patched_df.p', 'rb') as f:
    df = pickle.load(f)

with open('reg_model_att_def.p', 'rb') as f:
    reg_1, reg_2 = pickle.load(f)

with open('average_goals_dict.p', 'rb') as f:
    att_def_power = pickle.load(f)
# convert to strings
for key in att_def_power:
    att_def_power[key] = list(att_def_power[key])


df['Home_Advantage'] = df.apply(lambda x: 1 if x['Country_1'] in x['Location'] else -1 if x['Country_2'] in x['Location'] else 0, axis=1)

features = ['A1', 'D1', 'A2', 'D2', 'A1xA2',
            'D1xD2', 'A1xD2', 'D1xA2', 'Home_Advantage']

def make_x(a1, d1, a2, d2):
    return np.array([a1, d1, a2, d2, a1*a2, d1*d2, a1*d2, d1*a2, row['Home_Advantage']]).reshape(1, -1)

def predict(a1, d1, a2, d2):
    x = make_x(a1, d1, a2, d2)
    return reg_1.predict(x)[0], reg_2.predict(x)[0]

ADJUSTMENT_RATE = 0.021  # optimized number

# for i in range(0, 21):
for iter in range(20):
    # with open('average_goals_dict.p', 'rb') as f:
    #     att_def_power = pickle.load(f)
    # # convert to strings
    # for key in att_def_power:
    #     att_def_power[key] = list(att_def_power[key])


    total_error = 0
    # make sure dataframe is sorted in chronological order
    # create features on the fly
    for col in ['A1', 'D1', 'A2', 'D2']:
        df[col] = 0

    for i, row in df.iterrows():
        a1, d1 = att_def_power[row.Country_1]
        a2, d2 = att_def_power[row.Country_2]
        df.loc[i, 'A1'], df.loc[i, 'D1'] = a1, d1
        df.loc[i, 'A2'], df.loc[i, 'D2'] = a2, d2
        e1, e2 = predict(a1, d1, a2, d2)
        total_error += (row.Score_1 - e1) ** 2 + (row.Score_2 - e2) ** 2
        # update team power
        att_def_power[row.Country_1][0] += (row.Score_1 - e1) * ADJUSTMENT_RATE
        att_def_power[row.Country_1][1] += (row.Score_2 - e2) * ADJUSTMENT_RATE

        att_def_power[row.Country_2][0] += (row.Score_2 - e2) * ADJUSTMENT_RATE
        att_def_power[row.Country_2][1] += (row.Score_1 - e1) * ADJUSTMENT_RATE

        # if i % 100 == 0:
        #     print(i)
    print(total_error)

    df['A1xA2'] = df['A1'] * df['A2']
    df['D1xD2'] = df['D1'] * df['D2']
    df['A1xD2'] = df['A1'] * df['D2']
    df['D1xA2'] = df['D1'] * df['A2']

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



# nl_df = df[(df.Country_1 == 'Netherlands') | (df.Country_2 == 'Netherlands')]
# a1 = np.where(nl_df.Country_1 == 'Netherlands', nl_df.A1, nl_df.A2)
# d1 = np.where(nl_df.Country_1 == 'Netherlands', nl_df.D1, nl_df.D2)
#
# from matplotlib import pyplot as plt
# plt.plot(a1)
# plt.plot(d1)
# plt.show()
