# Functions for filtering an advertising data set
# Ernest Son
# Sam Chanow

import pandas as pd
from proximitypyhash import get_geohash_radius_approximation
from datetime import datetime as dt
from overpy import Overpass

def query_adid(adid, df):
    if adid == '' or df is None:
        return
    parsed_df = df.loc[df.advertiser_id == adid]
    return parsed_df

def query_location(lat, long, radius, df):
    # First we need to check that all the fields were filled out and then cast to floats
    if lat == '' or long == '' or radius == '':
        return None
    lat = float(lat)
    long = float(long)
    radius = float(radius)
    # We need to get the geohashes in the given radius at the given point
    geohashes = get_geohash_radius_approximation(
        latitude=lat,
        longitude=long,
        radius=radius,
        precision=10,
        georaptor_flag=False,
        minlevel=1,
        maxlevel=12)
    # Now create the new smaller dataframe
    parsed_df = df.loc[df.geohash.isin(geohashes)]
    return parsed_df

def query_date(start_date, start_time, end_date, end_time, df):
    # Combine dates and times together for optimal filtering
    start = dt.combine(start_date, start_time)
    end = dt.combine(end_date, end_time)
    if start_date == '' or end_date == '':
        return
    # convert string to datetime
    df['date'] = pd.to_datetime(df['datetime'])
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    # Parse the df by datetime
    date_filter = (df['date'] >= start) & (df['date'] <= end)
    parsed_df = df.loc[date_filter]
    print(parsed_df.head())
    parsed_df['datetime'] = parsed_df['datetime'].astype(str)
    print(parsed_df.head())
    print(len(parsed_df.index))
    return parsed_df

def query_node(lat, long, radius, node_name, df):
    api = Overpass()
    # First we need to check that all the fields were filled out and then cast to floats
    if lat == '' or long == '' or radius == '':
        return None
    lat = float(lat)
    long = float(long)
    radius = float(radius)
    # We need to get the nodes in the given radius at the given point
    # Now create the new smaller dataframe
    query = "node(around:" + str(radius) + ", " + str(lat) + ", " + str(long) + "); out body;"
    result = api.query(query)
    nodes = pd.DataFrame()
    for node in result.nodes:
        name = node.tags.get('name')
        node_lat = float(node.lat)
        node_lon = float(node.lon)
        if name == node_name or (node_name == '' and name is not None):
            node_df = pd.DataFrame([[name, node_lat, node_lon]], columns=['node_name', 'latitude', 'longitude'])
            nodes = pd.concat([nodes, node_df])
    nodes = nodes.reset_index(drop=True)
    return nodes

