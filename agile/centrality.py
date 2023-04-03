import pandas as pd
import numpy as np
import networkx as nx
from pygeohash import decode
from .filtering import query_location

def people_at_location(lat: float, long: float, radius: float, data: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame of all the people who visited a specified location."""

    filtered_data = query_location(lat, long, radius, data)

    adids = filtered_data.advertiser_id.unique()
    return data.loc[data.advertiser_id.isin(adids)]

def visited_locations(data: pd.DataFrame) -> np.ndarray:
    """Returns an array of all unique geohashes in the given DataFrame."""

    return data.geohash.unique()

def people_of_interest(data: pd.DataFrame) -> np.ndarray:
    """Returns an array of all unique advertiser IDs in the given DataFrame."""

    return data.advertiser_id.unique()

def compute_adjacency(row: pd.Series, A: np.ndarray, interest: np.ndarray) -> None:
    """Updates the adjacency matrix using the given row."""

    A[np.where(interest == row.advertiser_id)[0], np.where(interest == row.geohash)[0]] += 1

def centrality(people: np.ndarray, locations: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame of the centrality of each location given a list of people and places."""

    interest = np.concatenate((people, locations))
    A = np.zeros((len(interest), len(interest)))

    print(interest)

    data.apply(lambda row: compute_adjacency(row, A, interest), axis=1)

    G = nx.Graph(A)
    degree_centrality = nx.degree_centrality(G)
    centrality_values = [degree_centrality[i] for i in range(len(degree_centrality))]

    out_data = pd.DataFrame({'id': interest, 'centrality': centrality_values})
    out_data = out_data.iloc[len(people):]
    out_data[['latitude', 'longitude']] = pd.DataFrame(out_data['id'].apply(decode).tolist(), index=out_data.index)

    return out_data.sort_values(by='centrality', ascending=False).head()

def compute_top_centrality(lat: float, long: float, radius: float, N: int, data: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame of the top N most central locations given a latitude, longitude, radius, and DataFrame of data."""

    data_of_interest = people_at_location(lat, long, radius, data)

    visited = visited_locations(data_of_interest)
    people = people_of_interest(data_of_interest)

    return centrality(people, visited, data)

