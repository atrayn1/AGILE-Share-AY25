# Sam Chanow
# Ernest Son
# The algorithm for returning locations of interest

import pandas as pd
import pygeohash as gh
from datetime import datetime as dt

# Location of Interest Algorithm
# Prototype
# Input: Dataframe with Lat, Long, Geohash, Timestamp
# Assuming that that all of the points in the dataframe relate to the same ADID
# I.E. the dataframe has already been filtered
# Return: A filtered dataframe with the Locations of Interest
def LOI(data):

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

    # Sort by time
    data.loc[:, ('dates')] = pd.to_datetime(data['datetime']) # No setting with copy error
    data.sort_values(by="dates")

    # We should probably ensure that our geohashing is of sufficient precision anyways
    # We don't want to be too precise or else every data point will have its own geohash
    data["geohash"] = df.apply(lambda d : gh.encode(d.latitude, d.longitude, precision=4), axis=1)

    # Ensure the dataframe has a geohash column, otherwise we will geohash it
    # ourselves with a default range
    '''
    if 'geohash' not in data.columns:
        # geohash ourselves
        data["geohash"] = df.apply(
                lambda d : gh.encode(d.latitude, d.longitude, precision=10), axis=1
                )
    '''

    # With the geohashes this seems straight forward:
    # For every unique geohash we will repeat the search process
    # We are looking for two things, staying in one geohash for a long time, or
    # repeatedly coming back to a geohash over long periods of time

    # For long dwells this will be iterative so O(n) but I think we can upper
    # bound it there and I really do not think it will get any better than that
    # Actually we may be able to scrt this using Pandas Map but it may get janky

    # df.values can produce a 2d numpy array which for us may actually be the
    # best choice ... at least for a proof of concept

    # So at this point the dataframe needs to be exactly formated the same so
    # column indices are consistent

    data_values = data[['geohash', 'datetime']].values
    data_size = len(data_values)

    # Trying to keep O(n)
    for index in range(0, data_size):

        # Do not do anything on the first point
        if index == 0:
            continue
        
        # Now here is the meat and potatoes
        # We are looking for relationships between rows specifically:
            # Same Geohash over long period of time
        start_index = index #Keep track of where we started

        # Do While Loop emulation
        while True:
            # we reached the end of the array
            if index == data_size - 1:
                break

            # If the geohashes are not equal
            c_geohash = data_values[index, 0]
            n_geohash = data_values[index+1, 0]
            if c_geohash != n_geohash:
                break # Exit the while Loop
            
            index += 1 # Go to the next row

        # Now that we have broken out of the loop we compare the time from
        # start_index to index

        # Convert strings to datetime objects or integers or something
        start_time = dt.strptime(data_values[start_index, 1], '%Y-%m-%d %H:%M:%S') 
        end_time = dt.strptime(data_values[index, 1], '%Y-%m-%d %H:%M:%S')
        time_difference = end_time - start_time
        #print('start:', start_index, 'start_index:', index)
        #print('time difference:', time_difference.total_seconds())

        # TODO
        # If this is above a certain threshhold we mark it or throw it into a
        # new filtered dataframe

        # we could pretty-print data points at a location of interest with a
        # different color for the demo

        # 3600 seconds in an hour
        if time_difference.total_seconds() > 3600:
            print(data_values[start_index, 0], 'sus')

# testing
df = pd.read_csv("../data/one_id.csv")
LOI(df)
