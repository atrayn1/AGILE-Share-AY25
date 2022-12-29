# Ernest Son
# Prototype algorithm for identifying colocated devices

import numpy as np
import pandas as pd
import pygeohash as gh
from datetime import datetime as dt

# INPUT
#   data:
#     dataframe containing all of the data
#   lois:
#     dataframe of just locations of interest
#   precision:
#     geohashing precision value
# OUTPUT
#   data_out:
#     a dataframe containing co-located devices
def colocations(data, lois, precision, debug=False) -> pd.DataFrame:

    # This is the output dataframe, i.e. where we store suspicious data points.
    relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']
    data_out = pd.DataFrame(columns=relevant_features)

    # Sort by time
    data.loc[:, ('dates')] = pd.to_datetime(data['datetime'])
    data.sort_values(by="dates", inplace=True)

    # We ensure that our geohashing is of sufficient precision. We don't want to
    # be too precise or else every data point will have its own geohash.
    data["geohash"] = data.apply(lambda d : gh.encode(d.latitude, d.longitude, precision=precision), axis=1)


    # TODO
    # 1)
    # From the main data array, grab rows that have a geohash that is also found
    # within the locations of interest dataframe. Pandas should make this easy.
    # This produces a "filtered" dataframe that we will work with.
    loi_geohashes = lois['geohash'].unique()
    data_filtered = data[data['geohash'].isin(loi_geohashes)]
    # DEBUG
    if debug:
        print(data_filtered)
    data_values = data_filtered[relevant_features].values
    data_size = len(data_values)

    # 2)
    # Make a dictionary of all the adIDs.
    adid_dict = dict()

    # TODO
    # 3)
    # In the filtered dataframe, check a few things ...
    # ... different adID from the one associated with the geohash in "lois"?
    # ... timestamp is within the same timeframe as the identified LOI?
    # If all of these criteria are met, increment by one in the dictionary for
    # the adID of the datapoint that we're working on.
    # If, after the increment, the dictionary value exceeds a certain threshold,
    # then add that adID to data_out.
    for index in range(0, data_size):
        # Is the adid in the dictionary?
        if data_values[index, 4] in adid_dict:
            d_sus = pd.DataFrame(np.atleast_2d(data_values[index]), columns=relevant_features)
            data_out = pd.concat([data_out, d_sus], ignore_index=True)

    # Return the suspicious data points
    #return data_out

# testing
df = pd.read_csv("../../data/week.csv")
locations = pd.read_csv("../../data/lois.csv")
colocations(data=df, lois=locations, precision=10, debug=True)
