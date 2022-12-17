#Simple OverpassAPI Query wrappers for ease of use
#Sam Chanow

import overpy
import pandas as pd

'''
# Give a single latitude and longitude and range
# Return the named Overpass Nodes within the range from the point
def overpassNearbyQuery(latitude, longitude, range):
    api = overpy.Overpass()
    #Build the query
    query = "node(around:" + str(range) + ", " + str(latitude) + ", " + str(longitude) + "); out body;"
    
    # fetch all ways and nodes
    result = api.query(query)

    return filterNodeList(results=result)
'''

# Given a list of latitude and longitudes and a range
# Build a polygon that encompasses those points
# Return all the named Overpass Nodes within a range from that polygon
# Need to be LISTS of latitude and Longitude
# Range is in Meters
# Needs a list of dates so that the polyline is created chronologically

# UPDATE: Function now takes a dataframe 
# THIS DATAFRAME MUST CONTAIN 'latitude' 'longitude' 'datetime'

def overpassPolyLineNearbyQuery(data, range):
    # custom polygon boundary. Boundaries specified as 
    # (poly:lat1 long1 lat2 long2............latN longN lat1 long1). 
    # Coordinates (lat, long) at the beginning and at the end of the 
    # poly are the same signifying that it is a closed polygon. 
    # If there is a mismatch here, query fails with an error message.

    # To build the polygon I am thinking of looping through the list frontways and then backways.
    # Effectively creating a long line

    # TODO
    # WE NEED TO PROBABLY take in the DF AND SORT BY TIME SO IT IS AN ACCURATE POLYLINE###
    #data['dates'] = pd.to_datetime(data['datetime'])
    data.loc[:, ('dates')] = pd.to_datetime(data['datetime']) # No setting with copy error
    data.sort_values(by="dates")

    latitudes = data['latitude'].tolist()
    longitudes = data['longitude'].tolist()

    # Assuming the points are being passed in as a list
    api = overpy.Overpass()

    # Create a list of location pairs (as alist themselevs)
    points = [[str(pair[0]), str(pair[1])] for pair in zip(latitudes, longitudes)]

    # Reverse the list and then concat them together
    # points_r = list(reversed(points))
    # polygon = points + points_r
    # Deconstruct the polygon pairs so that they can be joined easily for the string query
    polyline_list = sum(points, [])

    #build the query
    query = "node(around:" + str(range) + ", " + ", ".join(polyline_list) + "); out body;"

    #fetch the results
    result = api.query(query)

    return filterNodeList(results=result)


#Given an overpass Query Result
#Return a dataframe of all the named nodes and their lat/longs
def filterNodeList(results):

    #Get the useful information (the nodes) and discard the rest
    results = results.nodes

    #Build the Dataframe with the names,lats,and longs
    nodes = pd.DataFrame()
    nodes['name'] = [node.tags.get("name", "n/a") for node in results]
    nodes['latitude'] = [float(node.lat) for node in results] 
    nodes['longitude'] = [float(node.lon) for node in results]

    #Filter out the n/a rows
    nodes = nodes[nodes['name'] != "n/a"]

    print(type(nodes))
    return nodes

#df = pd.read_csv(
#  "../data/test_location_data_gh.csv"
#)

#Test Code (Single Query)
#res = overpassNearbyQuery(df['latitude'][0], df['longitude'][0], 1000)
#print(res)


#TEST CODE ON A POLYLINE
#Fuction that querys a dataframe and filters based on AdId
#def query_adid(adid, df):
#    if adid == '':
#        return

    #Parse the df based on advertsing id
#    parsed_df = df.loc[df.advertiser_id == adid]
#    return parsed_df

#Test Code Polygon Query
#filtered_df = query_adid(df['advertiser_id'][0], df)
#res = overpassPolyLineNearbyQuery(filtered_df, 10)
#print(res)
