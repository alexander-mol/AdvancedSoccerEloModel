import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson

from matplotlib import pyplot as plt

import utils

with open('2010-2018_patched_df.p', 'rb') as f:
    df = pickle.load(f)

low = 1550
high = 1600
mask = (df.Elo_Score_Before_1 > low) & (df.Elo_Score_Before_1 < high) & (df.Elo_Score_Before_2 > low) & (df.Elo_Score_Before_2 < high)
scores = np.concatenate((df[mask]['Score_1'].values, df[mask]['Score_2'].values))


mean = np.average(scores)
print(mean)
# plt.hist(df.Elo_Score_Before_1)
# plt.show()

for i in range(10):
    print(i, np.sum(scores==i) / len(scores), poisson.pmf(i, mean))
