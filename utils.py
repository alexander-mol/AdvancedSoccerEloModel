import numpy as np
from scipy.stats import poisson
import re
import requests


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


def outcome_dict(e1, e2):
    p1, p2, pd = 0, 0, 0
    e1 = np.maximum(e1, 0)
    e2 = np.maximum(e2, 0)
    for i in range(20):
        for j in range(20):
            pm = poisson.pmf(i, e1) * poisson.pmf(j, e2)
            if i > j:
                p1 += pm
            elif i == j:
                pd += pm
            else:
                p2 += pm
    return {'Team 1': p1, 'Team 2': p2, 'Draw': pd}

def get_score_outcomes(e1, e2):
    score_outcomes = []
    for i in range(10):
        for j in range(10):
            score_outcomes.append((i, j, poisson.pmf(i, e1) * poisson.pmf(j, e2)))
    score_outcomes.sort(key=lambda x: -x[2])
    return score_outcomes

def brier_score(predicted_prob_for_outcome):
    return np.average(np.power(1-predicted_prob_for_outcome, 2))


def derivatives(error_func, params, index, sensitivity=0.0000001):
    y1 = error_func(params)
    params[index] += sensitivity
    y2 = error_func(params)
    params[index] += sensitivity
    y3 = error_func(params)
    d1 = (y2 - y1) / sensitivity
    d2 = (y3 - y2) / sensitivity
    return (d2 - d1) / sensitivity, (d1 + d2) / 2, y2


def optimize(error_func, starting_parameters, sensitivity=0.0000001, learning_rate=1):
    params = starting_parameters
    count = 0
    while True:
        for i in range(len(params)):
            d__, d_, error = derivatives(error_func, params, i, sensitivity)
            step = (- d_ / d__) * learning_rate
            params[i] += step
            count += 1
            print(f'{count}: error: {error} params: {params}')


def apply_regression(X, coefs):
    return np.dot(X, coefs[1:]) + coefs[0]

def get_elo(country):
    html = requests.get('https://en.wikipedia.org/wiki/World_Football_Elo_Ratings').text.lower()
    return int(re.findall(f'<td align="left"><span style="white-space:nowrap"><span class="flagicon"><img alt="" src=".*?" data-file-width=".+" data-file-height=".+" \/>.*?&#160;<\/span><a href=.*?">{country.lower()}<\/a><\/span><\/td>\n<td>(\d+)<\/td>', html)[0])
