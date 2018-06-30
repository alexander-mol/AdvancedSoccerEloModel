import re
import time
import pickle
from selenium import webdriver
import pandas as pd

t0 = time.time()
time_stamp = time.strftime('%Y%m%d-%H%M')

def get_html(url, find_element_by, value):
    browser = webdriver.Chrome('./chromedriver')
    browser.get(url)
    time.sleep(10)
    if find_element_by is None or value is None:
        data = browser.execute_script("return document.body.innerHTML")
    else:
        data = browser.find_element(find_element_by, value)
        data = data.get_attribute('innerHTML')
    browser.quit()
    return data


extractor = """<div class="slick-cell l0 r0 match-cell ">(.+?)<br>(.+?)<\/div><div class="slick-cell l1 r1 match-cell "><a href=".+?" onclick="handleLink\('.+?'\); return false;">(.+?)<\/a><br><a href=".+?" onclick="handleLink\('.+?'\); return false;">(.+?)<\/a><\/div><div class="slick-cell l2 r2 score-cell ">(.+?)<br>(.+?)<\/div><div class="slick-cell l3 r3 match-cell ">(.+?)<br><a href=".+?" onclick="handleLink\('.+?'\); return false;">(.+?)<\/a><\/div><div class="slick-cell l4 r4 change-cell ">(.+?)<br>(.+?)<\/div><div class="slick-cell l5 r5 score-cell ">(.+?)<br>(.+?)<\/div><div class="slick-cell l6 r6 change-cell ">(.+?)<br>(.+?)<\/div><div class="slick-cell l7 r7 score-cell ">(.+?)<br>(.+?)<\/div><\/div><div class="ui-widget-content slick-row .+?" style="top:.+?px">"""

def make_df(data):
    df = pd.DataFrame(data, columns=['Date', 'Year', 'Country_1', 'Country_2', 'Score_1', 'Score_2', 'Competition', 'Location', 'Elo_Score_Change_1', 'Elo_Score_Change_2', 'Elo_Score_New_1', 'Elo_Score_New_2', 'Rank_Change_1', 'Rank_Change_2', 'Rank_New_1', 'Rank_New_2'])
    int_cols = ['Score_1', 'Score_2', 'Elo_Score_Change_1', 'Elo_Score_Change_2', 'Elo_Score_New_1', 'Elo_Score_New_2', 'Rank_Change_1', 'Rank_Change_2', 'Rank_New_1', 'Rank_New_2']

    def to_int(x):
        if x[0] == '−':
            try:
                return -int(x[1:])
            except:
                return 0
        return int(x)

    for col in int_cols:
        df[col] = df[col].apply(to_int)
    return df

df_list = []
for year in range(2010, 2019):
    print(f'Getting data for {year}...')
    html = get_html(f'https://www.eloratings.net/{year}_results', 'class name', 'slick-viewport')
    data = re.findall(extractor, html)
    df_list.append(make_df(data))
df = pd.concat(df_list).reset_index(drop=True)

with open('2010-2018_match_df.p', 'wb') as f:
    pickle.dump(df, f)
