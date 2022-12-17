#Sam Chanow
# The algorithm for returning locations of interest

import pandas as pd

#Location of Interest Algorithm
#Prototype
#Input: Dataframe with Lat, Long, Geohash, Timestamp
#Assuming that that all of the points in the dataframe relate to the same ADID
#I.E. the dataframe has already been filtered
#Return: A filtered dataframe with the Locations of Interest
def LOI(data):

    #Now that we have locations sorted by time we can use iteration to view an ADIDs
    #Movement Chronologically

    #Probably the two biggest things that mark a locatio nof interest
    # 1) Extended stay in one location
    # 2) Repeated visits over extended period of time to one location

    #For this algorithm's efficiency we may have to lean very heavily on geohashing
    #Using specific precision geohashing would allow us to more easily see if a point moves from one general area
    #To another general area wihtout manual lat long calculations

    ###########################
    #  Algorithm Begins here  #
    ###########################

    #Sort by time
    data.loc[:, ('dates')] = pd.to_datetime(data['datetime']) # No setting with copy error
    data.sort_values(by="dates")

    #Ensure the dataframe has a geohash column, otherwise we will geohash it ourselves with a default range
    #TODO

    #With the Geohashes this seems straight forward
    #For every unique geohash we will repeat the search process
    #Again we are looking for two things, staying in one geohash for a long time, or repeatedly coming back to a geohash
    #Over long periods of time
