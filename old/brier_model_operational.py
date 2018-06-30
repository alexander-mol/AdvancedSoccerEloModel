import numpy as np

# long optimized coefs
coefs = np.array([ -1.44804123e-02, 9.69717241e-03, 9.22752006e-03, -1.92846927e-04, -5.26508671e-07, -1.99169373e-01, -2.08277502e-01, -6.97185005e-06, -4.07455680e-01, -2.20530669e-01, 1.91809715e-02, -4.65832723e-02, 9.54369609e-03, 9.86974336e-03, -7.61132959e-04, 8.56017445e-07, -2.74207199e-01, -2.68930190e-01, -6.78173539e-06, -5.43140510e-01, 1.03599439e-02, -2.15621714e-02])

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

print(expectation_scores(1800, 1800))