# Ernest Son
# Prototype algorithm for identifying colocated devices

import numpy as np
import pandas as pd
import pygeohash as gh
from datetime import datetime as dt

# ASSUMPTIONS
#   data is already geohashed to sufficient precision
# INPUT
#   data:
#     dataframe containing all of the data
#   lois:
#     dataframe of just locations of interest
# OUTPUT
#   data_out:
#     a dataframe containing co-located devices
def colocations(data, lois, precision, debug=False) -> pd.DataFrame:

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

    if debug:
        print("SORTED BY DATETIME:")
        print(data)
        print()

    # We ensure that our geohashing is of sufficient precision. We don't want to
    # be too precise or else every data point will have its own geohash.
    #data["geohash"] = data.apply(lambda d : gh.encode(d.latitude, d.longitude, precision=precision), axis=1)

    # 1)
    # From the main data array, grab rows that have a geohash that is also found
    # within the locations of interest dataframe. Pandas should make this easy.
    loi_geohashes = lois['geohash'].unique()
    data_filtered = data[data['geohash'].isin(loi_geohashes)]

    # 2)
    # Remove data points that come from the same advertiser_id as the LOI data
    # that we're working with.
    loi_adid = lois['advertiser_id'][0]
    data_filtered = data_filtered[data_filtered['advertiser_id'] != loi_adid]

    if debug:
        print("FILTERED DATA:")
        print(data_filtered)
        print()

    # TODO
    # 3)
    # From the filtered data points, are they there at the same time? Does the
    # timestamp associated with a given data point fall within a given range
    # around the first and last timestamp of the given location of interest?

    # Return the suspicious data points
    #return data_out

# testing
df = pd.read_csv("../../data/weeklong_gh.csv")
locations = pd.read_csv("../../data/lois.csv")
colocations(data=df, lois=locations, precision=10, debug=True)
