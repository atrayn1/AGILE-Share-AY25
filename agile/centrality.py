# Applying Graph Theory Centrality to ADIDs at a specific location

#TODO Update required libraries in docker files to include networkx as a dependency
import pandas as pd
import numpy as np
import networkx as nx
from pygeohash import decode
from filtering import query_location

# First step is identifying persons of interest and locations of interest
# This assumes that the file is already geohashed (which it should be)
def people_at_location(lat, long, radius, data) -> pd.DataFrame:
    #Basic filtering of points at a location
    filtered_data = query_location(lat, long, radius, data)

    # Get a list of unique adids that ppeared at this location
    adids = filtered_data['advertiser_id'].unique()

    # Return all of the data from the original dataframe of the peopel who viisted our 
    # specified location
    return data.loc[data.advertiser_id.isin(adids)]


# Data is a dataframe containing adids and geohashes
# Returns a list of all unique geohashes in data
def visited_locations(data) -> list: return data['geohash'].unique()

# Data is a dataframe that contains adids
# Returns a list of all unique adids in data
def people_of_interest(data) ->list: return data['advertiser_id'].unique()

# This funcion takes in a list of people and places (geohashes) and computes their centrality
# It also requires the full data to create the adjacency matrix
def centrality(people, locations, data) -> pd.DataFrame:

    print("P", people)
    print("L", locations)

    # Create interest list
    interest = np.concatenate((people, locations), axis=0)
    size = len(interest)

    #Create an empty adjacency matrix
    A = np.zeros([size, size])

    # Fill in adjacency matrix
    # We can do this easily with a janky apply function
    # I know that this is weird and I am against using global vars like this in a function,
    # But I cant think of any other way right now
    def compute_adjacency(row):
        # Not usre if this should be +1 or = 1 we will have to do some testing
        A[np.where(interest == row.advertiser_id)[0], np.where(interest == row.geohash)[0]] += 1

    print(A)

    # Take advantage of that pandas asynchronicity
    data.apply(compute_adjacency, axis=1)

    # Build the graph
    G = nx.Graph(A)

    #Calculate the degress of centrality
    degree_centrality = nx.degree_centrality(G)

    print(degree_centrality[0])

    # Degree Centrality is mapped from index to centrality so we need to pull it out
    centrality_values = [degree_centrality[i] for i in range(len(degree_centrality))]

    out_data = pd.DataFrame()
    out_data['id'] = interest
    out_data['centrality'] = centrality_values

    # Get rid of the data associated with the individuals and not the places
    return out_data.iloc[len(people):]

# Wrapper function to complete centrality computation from start to finish
def compute_top_centrality(lat, long, radius, N, data) -> pd.DataFrame:
    data_of_interest = people_at_location(lat, long, radius, data)
    visited = visited_locations(data_of_interest)
    people = people_of_interest(data_of_interest)

    out_data = centrality(people, visited, data)

    # Simple apply based wrapper for the pygeohash decode
    def decode_geohash(row):
        coord = decode(row['id'])
        row['latitude'] = coord[0]
        row['longitude'] = coord[1]
        return row

    out_data = out_data.apply(decode_geohash, axis=1)
    ordered_out_data = out_data.sort_values(by='centrality', ascending=False).head(N)

    return ordered_out_data


# Test code for centrality calculations

#data = pd.read_csv('../data/demo_2023-01-11.csv')

#lat = 46.2642
#lon = -119.2426
#range_m = 100

#centrality_data = compute_centrality(lat, lon, range_m, data)
#print(centrality_data)

#centrality_data_sorted = centrality_data.sort_values('centrality')

#print(centrality_data_sorted)