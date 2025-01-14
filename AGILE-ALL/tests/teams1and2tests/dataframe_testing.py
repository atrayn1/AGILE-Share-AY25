#Testing dataframe apply() and fitering for fast colocation

import numpy as np
import pandas as pd
import pygeohash as gh
from datetime import datetime, timedelta

# Test Dataframe
d = {
    'Name': ['Alisa', 'Bobby', 'jodha', 'jack', 'raghu', 'Cathrine',
             'Alisa', 'Bobby', 'kumar', 'Alisa', 'Alex', 'Cathrine'],
    'Age': [26, 24, 23, 22, 23, 24, 26, 24, 22, 23, 24, 24],

    'Score': [85, 63, 55, 74, 31, 77, 85, 63, 42, 62, 89, 77]}

df = pd.DataFrame(d, columns=['Name', 'Age', 'Score'])

# Test function to filter based on some mutation of the score

def score_mut(row):
    row['remove'] = row['Score'] < 70
    return row
#  parsed_df = df.loc[df.advertiser_id == adid]
df = df.apply(score_mut, axis=1)
parsed_df = df.loc[df['remove'] == False].drop(columns=['remove'])

print(parsed_df.head())