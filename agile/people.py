# Ernest Son
# Prototype algorithm for identifying colocated devices

import numpy as np
import pandas as pd
import pygeohash as gh
from datetime import datetime, timedelta

# TODO
# the idea is solid but there is room for improvement
# for a "real" data set you might get a ton of pings for a single location
# of interest, we should have more robust ways to distinguish somebody as a
# colocated actor

# ASSUMPTIONS
#   - data is already geohashed to sufficient precision
#   - locations of interest are from the same data set
# INPUT
#   data:
#     dataframe containing all of the data
#   lois:
#     dataframe of locations of interest for a single advertiser ID
#   timerange:
#     range in HOURS to match timeframes of datapoints
#     ex. "1" means within 1 hour before or within 1 hour after an LOI timestamp
# OUTPUT
#   data_out:
#     a dataframe containing co-located devices
def colocation(data, lois, duration, debug=False) -> pd.DataFrame:

    # This is the output dataframe, i.e. where we store suspicious data points.
    relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']
    data_out = pd.DataFrame(columns=relevant_features)

    # Fail gracefully if no LOIs exist
    if lois.empty:
        return data_out

    # Sort by time
    # with newer versions of pandas, we may need to use something this
    #data['datetime'] = pd.to_datetime(data['datetime'], infer_datetime_format=True)
    data['datetime'] = pd.to_datetime(data.datetime)
    data.sort_values(by="datetime", inplace=True)

    # filter only the useful columns
    data = data[relevant_features]
    data.reset_index(drop=True, inplace=True)

    # Ensure our geohashing is of sufficient precision.
    #data["geohash"] = data.apply(lambda d : gh.encode(d.latitude, d.longitude, precision=10), axis=1)

    # 1)
    # From the main data array, grab rows that have a geohash that is also found
    # within the locations of interest dataframe. Pandas should make this easy.
    loi_geohashes = lois.geohash.unique()
    filtered = data[data.geohash.isin(loi_geohashes)]

    # 2)
    # Remove data points that come from the same advertiser_id as the LOI data
    # that we're working with.
    loi_adid = lois.advertiser_id[0]
    filtered = filtered[filtered.advertiser_id != loi_adid]

    # 3)
    # From the filtered data points, are they there at the same time? Does the
    # timestamp associated with a given data point fall within a given range
    # around the first and last timestamp of the given location of interest?
    search_time = timedelta(hours=duration)

    # Return the row with a column (remove) if we should remove it
    def time_filter(row):
        loi_filtered = lois[lois.geohash == row.geohash]
        loi_dates = pd.to_datetime(loi_filtered.datetime, infer_datetime_format=True)
        filtered_time = row.datetime
        within_timerange = (filtered_time > (loi_dates - search_time)) & (filtered_time < (loi_dates + search_time))
        row['remove'] = not within_timerange.any()
        return row

    # Create the remove column
    filtered = filtered.apply(time_filter, axis=1)

    # Filter based on that column and drop it
    if 'remove' in filtered.columns:
        data_out = filtered.loc[filtered.remove == False].drop(columns=['remove'])

    # Return the suspicious data points
    if debug:
        print('colocated adIDs:')
        for id in data_out.advertiser_id.unique():
            print('-', id)
    return data_out

# testing
# to make weeklong_gh.csv and lois.csv I did the following steps:
# ebs@razer:../AGILE/tests$ python3 week.py
# ebs@razer:../AGILE/tests$ python3 geohash.py
# ebs@razer:../AGILE/agile/locations$ python3 loi.py

#df = pd.read_csv("../data/weeklong_gh.csv")
#locations = pd.read_csv("../data/lois.csv")
#colocation(data=df, lois=locations, duration=2, debug=True)

