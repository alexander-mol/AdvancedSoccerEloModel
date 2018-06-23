import re
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from scipy.stats import poisson

with open('2010-2018_match_df.p', 'rb') as f:
    df = pickle.load(f)

# apply patches
df['Competition'] = df['Competition'].apply(lambda x: re.sub('<.*?>', '', str(x)))

df['Elo_Score_Before_1'] = df['Elo_Score_New_1'] - df['Elo_Score_Change_1']
df['Elo_Score_Before_2'] = df['Elo_Score_New_2'] - df['Elo_Score_Change_2']

df['Outcome'] = df.apply(lambda x: 'T1' if x['Score_1'] > x['Score_2'] else 'draw' if x['Score_1'] == x['Score_2'] else 'T2', axis=1)
#
# with open('2010-2018_patched_df.p', 'wb') as f:
#     pickle.dump(df, f)

# need to randomize the 1st and 2nd team to avoid the bias introduced by always having the winning team in position 1
# unless the losing team is "home"
swap_12 = np.random.choice([True, False], len(df))
swap_columns = [col[:-2] for col in df.columns if '_1' in col]
for col in swap_columns:
    df.loc[swap_12, col+'_1'], df.loc[swap_12, col+'_2'] = df.loc[swap_12, col+'_2'], df.loc[swap_12, col+'_1']

# add features
df['Elo_Score_Diff'] = df['Elo_Score_Before_2'] - df['Elo_Score_Before_1']
df['Score_Diff^2'] = df['Elo_Score_Diff'] * df['Elo_Score_Diff']
df['LN(Elo_Score_Before_1)'] = np.log(df['Elo_Score_Before_1'])
df['LN(Elo_Score_Before_2)'] = np.log(df['Elo_Score_Before_2'])
df['Elo_Score_Product'] = df['Elo_Score_Before_1'] * df['Elo_Score_Before_2']
df['LN(Elo_Score_Product)'] = np.log(df['Elo_Score_Product'])
df['Friendly_Flag'] = (df['Competition'] == 'Friendly') * 1
df['Home_Advantage'] = df.apply(lambda x: 1 if x['Country_1'] in x['Location'] else -1 if x['Country_2'] in x['Location'] else 0, axis=1)

features = ['Elo_Score_Before_1', 'Elo_Score_Before_2', 'Elo_Score_Diff', 'Score_Diff^2', 'LN(Elo_Score_Before_1)',
            'LN(Elo_Score_Before_2)', 'Elo_Score_Product', 'LN(Elo_Score_Product)', 'Friendly_Flag', 'Home_Advantage']
# features = ['Elo_Score_Diff']


X = df[features]

y_1 = df['Score_1']
y_2 = df['Score_2']
y_c = df.apply(lambda x: 'T1' if x['Score_1'] > x['Score_2'] else 'draw' if x['Score_1'] == x['Score_2'] else 'T2', axis=1)


def print_coefs(reg):
    coefs = [round(coef, 3) for coef in reg.coef_]
    print(list(zip(features, coefs)))


def consolidated_outcome(e1, e2):
    p1, pd, p2 = np.zeros(len(e1)), np.zeros(len(e1)), np.zeros(len(e1))
    e1 = np.maximum(e1, 0)
    e2 = np.maximum(e2, 0)
    for i in range(10):
        for j in range(10):
            pm = poisson.pmf(i, e1) * poisson.pmf(j, e2)
            if i > j:
                p1 += pm
            elif i == j:
                pd += pm
            else:
                p2 += pm
    return p1, p2, pd


reg_1 = LinearRegression()
reg_1.fit(X, y_1)
print(reg_1.score(X, y_1))
print_coefs(reg_1)

reg_2 = LinearRegression()
reg_2.fit(X, y_2)
print(reg_2.score(X, y_2))
print_coefs(reg_2)

score_1_pred = reg_1.predict(X)
score_2_pred = reg_2.predict(X)
print(np.sum(np.power(df.Score_1 - score_1_pred, 2)) + np.sum(np.power(df.Score_2 - score_2_pred, 2)))

clf = LogisticRegression(C=100)
clf.fit(X, y_c)
print(clf.score(X, y_c), clf.classes_)
prob_pred = clf.predict_proba(X)
prob_of_outcome = np.zeros(len(prob_pred))
for i, outcome in enumerate(y_c):
    prob_of_outcome[i] = prob_pred[i][list(clf.classes_).index(outcome)]
print(f'Logistic CLF Brier score: {np.average(np.power(1-prob_of_outcome, 2))}')

e1 = reg_1.predict(X)
e2 = reg_2.predict(X)
probs = consolidated_outcome(e1, e2)
prob_of_outcome = np.zeros(len(prob_pred))
for i, outcome in enumerate(y_c):
    prob_of_outcome[i] = probs[['T1', 'T2', 'draw'].index(outcome)][i]
print(f'Reg model Brier score: {np.average(np.power(1-prob_of_outcome, 2))}')

for i in range(10):
    features = X.loc[i].values.reshape(1, -1)
    print(df['Score_1'][i],
          df['Score_2'][i],
          reg_1.predict(features),
          reg_2.predict(features),
          consolidated_outcome([reg_1.predict(features)[0]], [reg_2.predict(features)[0]]),
          clf.predict(features),
          clf.predict_proba(features))



def feature_usefulness(feature_name, X):
    reg_1 = LinearRegression()
    reg_1.fit(X, y_1)

    reg_2 = LinearRegression()
    reg_2.fit(X, y_2)
    score_before = reg_1.score(X, y_1) + reg_2.score(X, y_2)

    X_drop = X.drop(feature_name, 1)
    reg_1 = LinearRegression()
    reg_1.fit(X_drop, y_1)

    reg_2 = LinearRegression()
    reg_2.fit(X_drop, y_2)
    score_after = reg_1.score(X_drop, y_1) + reg_2.score(X_drop, y_2)
    return score_before - score_after


def least_useful_feature(X):
    feature_value = []
    for feature in X.columns:
        feature_value.append((feature, feature_usefulness(feature, X)))
    return min(feature_value, key=lambda x: x[1])


# X_copy = X.copy()
# while True:
#     print(list(X_copy.columns))
#     least_useful = least_useful_feature(X_copy)
#     print(f'Least useful: {least_useful}')
#     if least_useful[1] < 0.001:
#         X_copy.drop(least_useful[0], 1, inplace=True)
#     else:
#         break

# optimal_features = ['Elo_Score_Before_2', 'Elo_Score_Diff', 'Score_Diff^2', 'LN(Elo_Score_Before_1)',
#                     'LN(Elo_Score_Before_2)', 'Elo_Score_Product', 'Friendly_Flag', 'Home_Advantage']
# X = df[features]


with open('reg_model.p', 'wb') as f:
    pickle.dump((reg_1, reg_2), f)
