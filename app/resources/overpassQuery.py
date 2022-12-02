#Simple OverpassAPI Query wrappers for ease of use
#Sam Chanow

import overpy
import pandas as pd

#Give a single latitude and longitude and range
#Return the named Overpass Nodes within the range from the point
def overpassNearbyQuery(latitude, longitude, range):
    api = overpy.Overpass()
    #Build the query
    query = "node(around:" + str(range) + ", " + str(latitude) + ", " + str(longitude) + "); out body;"
    
    # fetch all ways and nodes
    result = api.query(query)

    return filterNodeList(results=result)

#Given a list of latitude and longitudes and a range
#Build a polygon that encompasses those points
#Return all th named Overpass Nodes within a range from that polygon
#Need to be LISTS of latitude and Longitude
#Range is in Meters
def overpassBoundingBoxNearbyQuery(latitudes, longitudes, range):
    #custom polygon boundary. Boundaries specified as 
    # (poly:lat1 long1 lat2 long2............latN longN lat1 long1). 
    # Coordinates (lat, long) at the beginning and at the end of the 
    # poly are the same signifying that it is a closed polygon. 
    # If there is a mismatch here, query fails with an error message.

    #To build the polygon I am thinking of looping through the list frontways and then backways.
    #Effectively creating a long line

    #Assuming the points are being passed in as a list
    api = overpy.Overpass()

    #Create a list of location pairs (as alist themselevs)
    points = [[str(pair[0]), str(pair[1])] for pair in zip(latitudes, longitudes)]

    #Reverse the list and then concat them together
    #points_r = list(reversed(points))
    #polygon = points + points_r
    #Deconstruct the polygon pairs so that they can be joined easily for the string query
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
    nodes['latitude'] = [node.lat for node in results] 
    nodes['longitude'] = [node.lon for node in results]

    #Filter out the n/a rows
    nodes = nodes[nodes['name'] != "n/a"]

    return nodes

#df = pd.read_csv(
#  "../data/test_location_data_gh.csv"
#)

#Test Code (Single Query)
#res = overpassNearbyQuery(df['latitude'][0], df['longitude'][0], 1000)
#print(res)

#Fuction that querys a dataframe and filters based on AdId
#def query_adid(adid, df):
#    if adid == '':
#        return

    #Parse the df based on advertsing id
#    parsed_df = df.loc[df.advertiser_id == adid]
#    return parsed_df

#Test Code Polygon Query
#filtered_df = query_adid(df['advertiser_id'][0], df)
#res = overpassBoundingBoxNearbyQuery(filtered_df['latitude'].tolist(), filtered_df['longitude'].tolist(), 10)
#print(res)