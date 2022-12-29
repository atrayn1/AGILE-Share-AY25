# Sam Chanow
# Ernest Son
# The algorithm for returning locations of interest

import numpy as np
import pandas as pd
from pygeohash import encode
from datetime import datetime as dt

# Location of Interest Algorithm
# Prototype
# Input: Dataframe w/ geohash, timestamp, latitude, longitude, and adID
#        geohashing precision value
#        length of an extended stay in hours
#        length to check for repeated visits
# Assuming that that all of the points in the dataframe relate to the same adID
# I.E. the dataframe has already been filtered
# Return: A filtered dataframe with the Locations of Interest
def locations_of_interest(data, precision, extended_duration, repeated_duration, debug=False) -> pd.DataFrame:

    # Now that we have locations sorted by time we can use iteration to view an
    # ADIDs movement Chronologically

    # Probably the two biggest things that mark a location of interest
    # 1) Extended stay in one location
    # 2) Repeated visits over extended period of time to one location

    # For this algorithm's efficiency we may have to lean very heavily on
    # geohashing

    # Using specific precision geohashing would allow us to more easily see if a
    # point moves from one general area to another general area without manual
    # lat/long calculations

    ###########################
    #  Algorithm Begins here  #
    ###########################

    # These are the features we care about in the input dataframe
    relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']

    # This is the output dataframe, i.e. where we store suspicious geocodes
    data_out = pd.DataFrame(columns=relevant_features)

    # Sort by time
    data.loc[:, ('dates')] = pd.to_datetime(data['datetime']) # No setting with copy error
    data.sort_values(by="dates", inplace=True) # Inplace fixes the time sorting error

    # We ensure that our geohashing is of sufficient precision. We don't want to
    # be too precise or else every data point will have its own geohash.
    data["geohash"] = data.apply(lambda d : encode(d.latitude, d.longitude, precision=precision), axis=1)

    # With the geohashes this seems straight forward:
    # For every unique geohash we will repeat the search process
    # We are looking for two things, staying in one geohash for a long time, or
    # repeatedly coming back to a geohash over long periods of time

    # For long dwells this will be iterative so O(n) but I think we can upper
    # bound it there and I really do not think it will get any better than that
    # Actually we may be able to scrt this using Pandas Map but it may get janky

    # df.values can produce a 2d numpy array which for us may actually be the
    # best choice ... at least for a proof of concept

    # So at this point the dataframe needs to be exactly formatted the same so
    # column indices are consistent

    data_values = data[relevant_features].values
    data_size = len(data_values)

    # 1) Extended stay in one location
    # In other words, we identify adjacent rows with the same geohash
    # Trying to keep O(n)
    for index in range(0, data_size):

        # Do not do anything on the first point
        if index == 0:
            continue
        
        # Now here is the meat and potatoes
        # We are looking for relationships between rows specifically:
        # Same Geohash over long period of time
        start_index = index # Keep track of where we started

        # Do-While-Loop emulation
        while True:
            # We reached the end of the array
            if index == data_size - 1:
                break
            # If the geohashes are not equal
            c_geohash = data_values[index, 0]
            n_geohash = data_values[index+1, 0]
            if c_geohash != n_geohash:
                break # Exit the while Loop
            # Go to the next row
            index += 1

        # Now that we have broken out of the loop we compare the timestamps of
        # start_index and end_index, if we exceed some specific threshold we add
        # relevant rows to the output dataframe
        end_index = index

        # Convert strings to datetime objects so we can compare them easily
        start_time = dt.strptime(data_values[start_index, 1], '%Y-%m-%d %H:%M:%S') 
        end_time = dt.strptime(data_values[end_index, 1], '%Y-%m-%d %H:%M:%S')
        time_difference = end_time - start_time
        if time_difference.total_seconds() > 3600 * extended_duration:

            # the centroid idea sucked, let's do a median data point based on
            # timestamp instead
            middle_index = (start_index + end_index) // 2

            # I know this is very wonky but the array shape was being weird
            # This does technically work but lets find a better solution
            '''
            d_sus = pd.DataFrame(columns=relevant_features)
            d_sus['geohash'] = [data_values[middle_index, 0]]
            d_sus['datetime'] = [data_values[middle_index, 1]]
            d_sus['latitude'] = [data_values[middle_index, 2]]
            d_sus['longitude'] = [data_values[middle_index, 3]]
            d_sus['advertiser_id'] = [data_values[middle_index, 4]]
            '''

            # Possibly a better solution
            d_sus = pd.DataFrame(np.atleast_2d(data_values[middle_index]), columns=relevant_features)
            data_out = pd.concat([data_out, d_sus], ignore_index=True)

    # DEBUG
    if debug:
        print('extended stays:')
        print(data_out.nunique())
        print('unique geohashes:', data_out['geohash'].unique())
        print()

    # 2) Repeated visits over extended period of time to one location

    # We need to look for repeated visits i.e. visits on multiple days
    # I am thinking we have a dictionary with the geohashes and when we see a geohash
    # we add it as key of dictionary where value is the timestamp
    # Then we can check time differences (more than 16 hours i.e. multiple visits) and still keep O(n)
    # This will make it O(2n)
    geohash_dict = dict()

    for index in range(0, data_size):
        # Is the geohash key in the dictionary?
        if data_values[index, 0] in geohash_dict:
            # Compare time in dict to current time
            # If time difference is >4 hours, add to final output dataframe
            start_time = dt.strptime(geohash_dict[data_values[index, 0]], '%Y-%m-%d %H:%M:%S') 
            end_time = dt.strptime(data_values[index, 1], '%Y-%m-%d %H:%M:%S')
            time_difference = end_time - start_time
            # 4 hours
            if time_difference.total_seconds() > 3600 * repeated_duration:
                '''
                d_sus = pd.DataFrame(columns=relevant_features)
                d_sus['geohash'] = [data_values[index, 0]]
                d_sus['datetime'] = [data_values[index, 1]]
                d_sus['latitude'] = [data_values[index, 2]]
                d_sus['longitude'] = [data_values[index, 3]]
                d_sus['advertiser_id'] = [data_values[index, 4]]
                '''
                d_sus = pd.DataFrame(np.atleast_2d(data_values[index]), columns=relevant_features)
                data_out = pd.concat([data_out, d_sus], ignore_index=True)
        
        geohash_dict[data_values[index, 0]] = data_values[index, 1]

    # DEBUG
    if debug:
        print('repeated visits:')
        print(data_out.nunique())
        print('unique geohashes:', data_out['geohash'].unique())
        print()

    # Remove duplicate geohashes so we limit the size of LOI list
    return data_out#.drop_duplicates(subset=['geohash'])

# testing
df = pd.read_csv("../../data/_54aa7153-1546-ce0d-5dc9-aa9e8e371f00_weeklong_gh.csv")
lois = locations_of_interest(data=df, precision=10, extended_duration=8, repeated_duration=24, debug=True)
lois.to_csv('lois.csv')

