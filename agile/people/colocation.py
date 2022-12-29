# Ernest Son
# Prototype algorithm for identifying colocated devices

import numpy as np
import pandas as pd
import pygeohash as gh
from datetime import datetime, timedelta

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
def colocation(data, lois, timerange, debug=False) -> pd.DataFrame:

    # This is the output dataframe, i.e. where we store suspicious data points.
    relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']
    data_out = pd.DataFrame(columns=relevant_features)

    # Sort by time
    # in the future, we may need to use something this
    #data['datetime'] = pd.to_datetime(data['datetime'], infer_datetime_format=True)
    data.loc[:, ('dates')] = pd.to_datetime(data['datetime'])
    data.sort_values(by="dates", inplace=True)

    # filter only the useful columns
    data = data[relevant_features]
    data.reset_index(drop=True, inplace=True)

    # We ensure that our geohashing is of sufficient precision. We don't want to
    # be too precise or else every data point will have its own geohash.
    #data["geohash"] = data.apply(lambda d : gh.encode(d.latitude, d.longitude, precision=10), axis=1)

    # 1)
    # From the main data array, grab rows that have a geohash that is also found
    # within the locations of interest dataframe. Pandas should make this easy.
    loi_geohashes = lois['geohash'].unique()
    filtered = data[data['geohash'].isin(loi_geohashes)]

    # 2)
    # Remove data points that come from the same advertiser_id as the LOI data
    # that we're working with.
    loi_adid = lois['advertiser_id'][0]
    filtered = filtered[filtered['advertiser_id'] != loi_adid]

    # TODO
    # 3)
    # From the filtered data points, are they there at the same time? Does the
    # timestamp associated with a given data point fall within a given range
    # around the first and last timestamp of the given location of interest?
    hours = timedelta(hours=timerange)
    filtered_values = filtered[relevant_features].values
    filtered_size = len(filtered_values)
    loi_values = lois[relevant_features].values
    loi_size = len(loi_values)
    for i in range(0, filtered_size):
        for j in range(0, loi_size):
            if filtered_values[i,0] == loi_values[j,0]:
                filtered_time = datetime.strptime(filtered_values[i,1], '%Y-%m-%d %H:%M:%S')
                loi_time = datetime.strptime(loi_values[j,1], '%Y-%m-%d %H:%M:%S')
                lower = loi_time - hours
                upper = loi_time + hours
                if filtered_time > lower or filtered_time < upper:
                    d_sus = pd.DataFrame(np.atleast_2d(filtered_values[i]), columns=relevant_features)
                    data_out = pd.concat([data_out, d_sus], ignore_index=True)

    # Return the suspicious data points
    if debug:
        print('colocated devices:')
        print(data_out['advertiser_id'].unique())
    return data_out

# testing
df = pd.read_csv("../../data/weeklong_gh.csv")
locations = pd.read_csv("../../data/lois.csv")
colocation(data=df, lois=locations, timerange=1, debug=True)
