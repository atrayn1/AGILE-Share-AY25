# Simple Script to add geohashes to test_location Dataset
# Sam Chanow

import pandas as pd
import numpy as np

data = pd.read_csv('../data/weeklong_gh.csv')
relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']
data = data[relevant_features]
data['datetime'] = pd.to_datetime(data.datetime)
data.sort_values(by='datetime', ascending=True, inplace=True)
data.sort_values(by='advertiser_id', inplace=True)
adids = data.groupby((data.advertiser_id != data.advertiser_id.shift()).cumsum())
for idx, adid in adids:
    print('time differences information for ' + adid.iloc[0].advertiser_id + ':')
    sorted_adid = adid.sort_values(by='datetime', ascending=True)
    sorted_adid['time_difference'] = sorted_adid.datetime.diff().fillna(pd.Timedelta(seconds=0))
    print('  max =', sorted_adid.time_difference.max())
    print('  median (sampling rate) =', sorted_adid.time_difference.median())
print()
