import numpy as np

# long optimized coefs
coefs = [-3.20889907e-02, 1.10602436e-02, 1.26205546e-02, -3.74474669e-03, -5.94367708e-06, -2.79859775e-01,
         -2.83783138e-01, -8.36086136e-06, -5.63650882e-01, -9.45846565e-02, -8.37924943e-01, -6.82393779e-02,
         1.32464843e-02, 1.29986712e-02, 4.68819670e-03, -6.54666855e-06, -4.05330334e-01, -4.08653704e-01,
         -8.81567097e-06, -8.13978862e-01, 7.15892934e-02, 7.69420643e-01]
mid = int(len(coefs) / 2)


def apply_regression(X, coefs):
    return np.dot(X, coefs[1:]) + coefs[0]


def expectation_scores(elo1, elo2, home_advantage_flag=False, friendly_flag=False):
    # ['Elo_Score_Before_1', 'Elo_Score_Before_2', 'Elo_Score_Diff', 'Score_Diff^2', 'LN(Elo_Score_Before_1)',
     # 'LN(Elo_Score_Before_2)', 'Elo_Score_Product', 'LN(Elo_Score_Product)', 'Friendly_Flag', 'Home_Advantage']
    features = np.array(
        [elo1, elo2, elo2 - elo1, (elo2 - elo1) ** 2, np.log(elo1), np.log(elo2), elo1 * elo2, np.log(elo1*elo2), friendly_flag * 1,
         home_advantage_flag * 1])
    e1 = apply_regression(features, coefs[:mid])
    e2 = apply_regression(features, coefs[mid:])
    return e1, e2

print(expectation_scores(1000, 1000))