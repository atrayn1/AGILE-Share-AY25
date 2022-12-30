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
def colocation(data, lois, hours, minutes, debug=False) -> pd.DataFrame:

    # This is the output dataframe, i.e. where we store suspicious data points.
    relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']
    data_out = pd.DataFrame(columns=relevant_features)

    # Sort by time
    # with newer versions of pandas, we may need to use something this
    #data['datetime'] = pd.to_datetime(data['datetime'], infer_datetime_format=True)
    data.loc[:, ('dates')] = pd.to_datetime(data['datetime'])
    data.sort_values(by="dates", inplace=True)

    # filter only the useful columns
    data = data[relevant_features]
    data.reset_index(drop=True, inplace=True)

    # Ensure our geohashing is of sufficient precision.
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

    # 3)
    # From the filtered data points, are they there at the same time? Does the
    # timestamp associated with a given data point fall within a given range
    # around the first and last timestamp of the given location of interest?
    hours = timedelta(hours=hours)
    minutes = timedelta(minutes=minutes)
    search_time = hours + minutes
    filtered_values = filtered[relevant_features].values
    filtered_size = len(filtered_values)
    '''
    loi_values = lois[relevant_features].values
    loi_size = len(loi_values)

    # convert this to use pd.DataFrame().apply()
    # I don't think we even need to use apply I think we can probably use
    # A strategy as showed on this thread https://stackoverflow.com/questions/51589573/pandas-filter-data-frame-rows-by-function
    # We can create a function to return a bool
    for i in range(0, filtered_size):
        for j in range(0, loi_size):
            filtered_geohash = filtered_values[i,0]
            loi_geohash = loi_values[j,0]
            if filtered_geohash == loi_geohash:
                filtered_time = datetime.strptime(filtered_values[i,1], '%Y-%m-%d %H:%M:%S')
                loi_time = datetime.strptime(loi_values[j,1], '%Y-%m-%d %H:%M:%S')
                lower = loi_time - search_time
                upper = loi_time + search_time
                if (filtered_time > lower) and (filtered_time < upper):
                    d_sus = pd.DataFrame(np.atleast_2d(filtered_values[i]), columns=relevant_features)
                    data_out = pd.concat([data_out, d_sus], ignore_index=True)
    '''

    # Making strides towards a better solution
    # The body of this for-loop should be trivial to apply()
    '''
    for i in range(0, filtered_size):
        filtered_geohash = filtered_values[i,0]
        loi_filtered = lois[lois['geohash'] == filtered_geohash]
        loi_dates = pd.to_datetime(loi_filtered.datetime, infer_datetime_format=True)
        filtered_time = datetime.strptime(filtered_values[i,1], '%Y-%m-%d %H:%M:%S')
        within_timerange = (filtered_time > (loi_dates - search_time)) & (filtered_time < (loi_dates + search_time))
        if within_timerange.any():
            d_sus = pd.DataFrame(np.atleast_2d(filtered_values[i]), columns=relevant_features)
            data_out = pd.concat([data_out, d_sus], ignore_index=True)
    '''

    # TODO
    # Just need to return the row properly so we can concat it to data_out
    # Return the rwo with a column (remove) if we should remove it
    def time_filter(row):
        loi_filtered = lois[lois.geohash == row.geohash]
        loi_dates = pd.to_datetime(loi_filtered.datetime, infer_datetime_format=True)
        filtered_time = datetime.strptime(row.datetime, '%Y-%m-%d %H:%M:%S')
        within_timerange = (filtered_time > (loi_dates - search_time)) & (filtered_time < (loi_dates + search_time))
        #if within_timerange.any():
            #print(row)
            #data_out = pd.concat([data_out, d_sus], ignore_index=True)
        row['remove'] = not within_timerange.any()
        return row
    #Create the remove column
    filtered = filtered.apply(time_filter, axis=1)
    #Filter based on that column and drop it
    data_out = filtered.loc[filtered['remove'] == False].drop(columns=['remove'])

    # Return the suspicious data points
    if debug:
        print('colocated devices:')
        print(data_out['advertiser_id'].unique())
    return data_out

# testing
# to make weeklong_gh.csv and lois.csv I did the following steps:
# ebs@razer:../AGILE/tests$ python3 week.py
# ebs@razer:../AGILE/tests$ python3 geohash.py
# ebs@razer:../AGILE/agile/locations$ python3 loi.py
df = pd.read_csv("../../data/weeklong_gh.csv")
locations = pd.read_csv("../../data/lois.csv")
colocation(data=df, lois=locations, hours=1, minutes=30, debug=True)
