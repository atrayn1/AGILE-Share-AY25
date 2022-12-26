#Sam Chanow
#Testing pandas Dataframe time sorting

import numpy as np
import pandas as pd
import pygeohash as gh
from datetime import datetime as dt

data = pd.read_csv("../data/_54aa7153-1546-ce0d-5dc9-aa9e8e371f00_weeklong.csv", sep=",")

# This is the output dataframe, i.e. where we store suspicious geocodes
data_out = pd.DataFrame(columns=['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id'])

# Sort by time
#format 2018-04-15 12:23:09
data.loc[:, ('dates')] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S') # No setting with copy error
data.sort_values(by="dates", inplace=True)

print(data['datetime'])

# We ensure that our geohashing is of sufficient precision. We don't want to
# be too precise or else every data point will have its own geohash.
data["geohash"] = data.apply(lambda d : gh.encode(d.latitude, d.longitude, precision=4), axis=1)

print(data.head())

relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']
data_values = data[relevant_features].values
latlongs = data[['latitude', 'longitude']].values
data_size = len(data_values)

    # 1) Extended stay in one location
    # In other words, we identify adjacent rows with the same geohash
    # Trying to keep O(n)
for index in range(0, data_size):

    # Convert strings to datetime objects so we can compare them easily
    start_time = dt.strptime(data_values[index, 1], '%Y-%m-%d %H:%M:%S') 
    end_time = dt.strptime(data_values[index+1, 1], '%Y-%m-%d %H:%M:%S')
    time_difference = end_time - start_time
    if time_difference.total_seconds() > 3600 or time_difference.total_seconds() < 0:
        print(time_difference)