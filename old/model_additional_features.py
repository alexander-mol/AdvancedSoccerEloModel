import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression


import utils

with open('2010-2018_patched_df.p', 'rb') as f:
    df = pickle.load(f)


def get_data_from_column(select_on, where_value, years=('2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018')):
    return df[(df[select_on] == where_value) & (df['Year'].isin(years))]

def get_country_goals_data(country):
    f1 = get_data_from_column('Country_1', country)['Score_1']  # goals for p1
    f2 = get_data_from_column('Country_2', country)['Score_2']  # goals for p2
    avg_goals_for = (f1.sum() + f2.sum()) / (f1.count() + f2.count())
    a1 = get_data_from_column('Country_1', country)['Score_2']  # goals against p1
    a2 = get_data_from_column('Country_2', country)['Score_1']  # goals against p2
    avg_goals_against = (a1.sum() + a2.sum()) / (a1.count() + a2.count())
    return avg_goals_for, avg_goals_against, f1.count() + f2.count()

# create average goals for and against dict by country including default value for small countries
default_val = df[df['Year'].isin(['2016', '2017', '2018'])].Score_1.mean()
countries = set(list(df['Country_1'].unique()) + list(df['Country_2'].unique()))
avg_goals_dict = {}
for country in countries:
    avg_goals_data = get_country_goals_data(country)
    if avg_goals_data[2] > 12:
        avg_goals_dict[country] = (avg_goals_data[0], avg_goals_data[1])
    else:
        avg_goals_dict[country] = (default_val, default_val)

# add features
df['Elo_Score_Diff'] = df['Elo_Score_Before_2'] - df['Elo_Score_Before_1']

df['Score_Diff^2'] = df['Elo_Score_Diff'] * df['Elo_Score_Diff']
df['LN(Elo_Score_Before_1)'] = np.log(df['Elo_Score_Before_1'])
df['LN(Elo_Score_Before_2)'] = np.log(df['Elo_Score_Before_2'])
df['Elo_Score_Product'] = df['Elo_Score_Before_1'] * df['Elo_Score_Before_2']
df['Friendly_Flag'] = (df['Competition'] == 'Friendly') * 1
df['Home_Advantage'] = df.apply(lambda x: 1 if x['Country_1'] in x['Location'] else -1 if x['Country_2'] in x['Location'] else 0, axis=1)
df['Home_AdvantagexElo_Score_Diff'] = df['Home_Advantage'] * df['Elo_Score_Diff']
df['Avg_For_1'] = df['Country_1'].apply(lambda x: avg_goals_dict[x][0])
df['Avg_Against_1'] = df['Country_1'].apply(lambda x: avg_goals_dict[x][1])
df['Avg_For_2'] = df['Country_2'].apply(lambda x: avg_goals_dict[x][0])
df['Avg_Against_2'] = df['Country_2'].apply(lambda x: avg_goals_dict[x][1])
df['Avg_For_1xAvg_For_2'] = df['Avg_For_1'] * df['Avg_For_2']
df['Avg_Against_1xAvg_Against_2'] = df['Avg_Against_1'] * df['Avg_Against_2']
df['Avg_For_1xAvg_Against_2'] = df['Avg_For_1'] * df['Avg_Against_2']
df['Avg_Against_1xAvg_For_2'] = df['Avg_Against_1'] * df['Avg_For_2']

# features = ['Elo_Score_Before_1', 'Elo_Score_Before_2', 'Elo_Score_Diff', 'Score_Diff^2', 'LN(Elo_Score_Before_1)',
#             'LN(Elo_Score_Before_2)', 'Elo_Score_Product', 'Friendly_Flag', 'Home_Advantage',
#             'Home_AdvantagexElo_Score_Diff',
#             'Avg_For_1', 'Avg_Against_1', 'Avg_For_2', 'Avg_Against_2', 'Avg_For_1xAvg_For_2',
#             'Avg_Against_1xAvg_Against_2', 'Avg_For_1xAvg_Against_2', 'Avg_Against_1xAvg_For_2']
features = ['Avg_For_1', 'Avg_Against_1', 'Avg_For_2', 'Avg_Against_2', 'Avg_For_1xAvg_For_2',
            'Avg_Against_1xAvg_Against_2', 'Avg_For_1xAvg_Against_2', 'Avg_Against_1xAvg_For_2', 'Home_Advantage']

X = df[features]
y_1 = df.Score_1
y_2 = df.Score_2

reg_1 = LinearRegression()
reg_1.fit(X, y_1)
print(reg_1.score(X, y_1))
reg_2 = LinearRegression()
reg_2.fit(X, y_2)
print(reg_2.score(X, y_2))

se_1 = np.sqrt(np.average((reg_1.predict(X) - df.Score_1) ** 2))
se_2 = np.sqrt(np.average((reg_2.predict(X) - df.Score_2) ** 2))
print(se_1, se_2)

print('\n')
for i in range(20):
    x = X.loc[i].values.reshape(1, -1)
    print(df['Score_1'][i],
          df['Score_2'][i],
          reg_1.predict(x),
          reg_2.predict(x))

print('\n')
for i, feature in enumerate(features):
    print(feature, reg_1.coef_[i], reg_2.coef_[i])
#
# with open('reg_model_att_def.p', 'wb') as f:
#     pickle.dump((reg_1, reg_2), f)
# #
# with open('average_goals_dict.p', 'wb') as f:
#     pickle.dump(avg_goals_dict, f)

# temp
for_1 = 1.810
against_1 = 0.709
for_2 = 1.238
against_2 = 0.896
x = np.array([for_1, against_1, for_2, against_2, for_1*for_2, against_1*against_2, for_1*against_2, against_1*for_2, 0]).reshape(1, -1)
print(reg_1.predict(x), reg_2.predict(x))