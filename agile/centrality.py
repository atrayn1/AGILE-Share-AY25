# Applying Graph Theory Centrality to ADIDs at a specific location

#TODO Update required libraries in docker files to include networkx as a dependency
import pandas as pd
import numpy as np
import networkx as nx
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
def people(data) ->list: return data['advertiser_id'].unique()

# This funcion takes in a list of people and places (geohashes) and computes their centrality
# It also requires the full data to create the adjacency matrix
def centrality(people, locations, data) -> pd.DataFrame:

    # Create interest list
    interest = people + locations
    size = len(interest)

    #Create an empty adjacency matrix
    A = np.zeros([size, size])

    # Fill in adjacency matrix
    # We can do this easily with a janky apply function
    # I know that this is weird and I am against using global vars like this in a function,
    # But I cant think of any other way right now
    def compute_adjacency(row):
        A[interest.index(row.advertiser_id), interest.index(row.geohash)] = 1

    # Take advantage of that pandas asynchronicity
    data.apply(compute_adjacency)

    # Build the graph
    G = nx.Graph(A)

    #Calculate the degress of centrality
    degree_centrality = nx.degree_centrality(G)

    out_data = pd.DataFrame()
    out_data['id'] = interest
    out_data['centrality'] = degree_centrality

    return out_data

# Wrapper function to complete centrality computation from start to finish
def compute_centrality(lat, long, radius, data) -> pd.DataFrame:
    pass