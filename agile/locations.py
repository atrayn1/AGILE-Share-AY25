# Sam Chanow
# Ernest Son
# Locations of Interest Algorithm Prototype

import numpy as np
import pandas as pd
from pygeohash import encode
from datetime import timedelta
from utils.geocode import reverse_geocode

# Input:  Dataframe w/ geohash, timestamp, latitude, longitude, and ad_id
#         Geohashing precision value
#         Length of an extended stay in hours
#         Length of minimum time between repeated visits
# Output: A dataframe containing all datapoints at locations of interest
def locations_of_interest(data, ad_id, precision, extended_duration, repeated_duration, debug=False) -> pd.DataFrame:

    # These are the features we care about in the input dataframe
    relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']
    data = data[relevant_features]

    # This is the output dataframe, i.e. where we store suspicious geocodes
    data_out = pd.DataFrame(columns=relevant_features)

    # Sort so we only have the ad_ids we care about
    data = pd.DataFrame(data[data.advertiser_id == ad_id])

    # THIS TAKES A LONG TIME ON LARGER DATA SETS
    # We ensure that our geohashing is of sufficient precision. We don't want to
    # be too precise or else every data point will have its own geohash.
    #data["geohash"] = data.apply(lambda d : encode(d.latitude, d.longitude, precision=precision), axis=1)

    # Sort by time and geohash
    # We need to do this in order to group adjacent geohashes properly
    data['datetime'] = pd.to_datetime(data.datetime)
    data.sort_values(by=['datetime', 'geohash'], ascending=[True,True], inplace=True)

    # 1) Extended stay in one location
    # In other words, we identify adjacent rows with the same geohash where
    # the difference between the maximum and minimum timestamp in a geohash
    # "group" is greater than some minimum "stay time" that we define as a
    # parameter to locations_of_interest()
    stay_groups = data.groupby((data.geohash != data.geohash.shift()).cumsum())
    extended_stays = stay_groups.agg({"datetime" : ["min", "max"]})
    extended_stays.columns = extended_stays.columns.droplevel(0)
    min_stay = timedelta(hours=extended_duration)
    extended_stays = extended_stays[min_stay < (extended_stays['max'] - extended_stays['min'])]
    for i in extended_stays.index:
        stay = stay_groups.get_group(i)
        data_out = pd.concat([data_out, stay], ignore_index=True)

    # DEBUG
    if debug:
        print('extended stays:', data_out.shape[0])

    # 2) Repeated visits over extended period of time to one location
    # We need to look for repeated visits i.e. visits on multiple days
    # I am thinking we have a dictionary with the geohashes and when we see a geohash
    # we add it as key of dictionary where value is the timestamp
    # Then we can check time differences (more than 16 hours i.e. multiple visits) and still keep O(n)
    # This will make it O(2n)
    geohash_dict = dict()
    def repeated_visits(row):
        # Is the geohash key in the dictionary?
        if row.geohash in geohash_dict:
            start_time = geohash_dict[row.geohash]
            end_time = row.datetime
            time_difference = end_time - start_time
            search_time = timedelta(hours=repeated_duration)
            within_timerange = time_difference > search_time
            row['remove'] = not within_timerange
        geohash_dict[row.geohash] = row.datetime
        return row
    data = data.apply(repeated_visits, axis=1)
    repeated_visits_df = data.loc[data.remove == False].drop(columns=['remove'])
    data_out = pd.concat([data_out, repeated_visits_df], ignore_index=True)

    # DEBUG
    if debug:
        print('repeated visits:', repeated_visits_df.shape[0])

    # Make sure we only get the columns we want
    data_out = data_out[relevant_features]

    # Reverse-geocode all the data points
    data_out = reverse_geocode(data_out)

    if debug:
        print('final dataframe:')
        print(data_out)
    return data_out

# TODO
# Labeling locations of interest will probably require the use of the Overpass
# API in order to distinguish the types of geographic nodes (residences,
# offices, malls, etc.)

# testing
#df = pd.read_csv("../data/weeklong_gh.csv")
#ubl = "54aa7153-1546-ce0d-5dc9-aa9e8e371f00"
#lois = locations_of_interest(data=df, ad_id=ubl, precision=10, extended_duration=7, repeated_duration=24, debug=True)

