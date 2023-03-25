# Functions for filtering an advertising data set
# Ernest Son
# Sam Chanow

import pandas as pd
from proximitypyhash import get_geohash_radius_approximation
from datetime import datetime as dt
import requests

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

# Find named nodes within a specified radius of a given latitude and longitude.
def query_node(lat, lon, rad, name):

    # Check that all fields were filled out and convert to floats
    if not all([lat, lon, rad]):
        return None
    lat, lon, rad = map(float, [lat, lon, rad])

    # Define query to look for named nodes within the specified radius
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json];
    node(around:{rad},{lat},{lon})[name];
    out center;
    """
    query = overpass_query.format(rad=rad, lat=lat, lon=lon)

    # Send the query to the Overpass API
    response = requests.get(overpass_url, params={'data': query})

    # Raises exception when not a 2xx response
    response.raise_for_status()

    # Parse the JSON response
    data = response.json()
    df = pd.json_normalize(data['elements'])

    # Filter the dataframe to only include named nodes
    df = df[df['tags.name'].notnull()]

    # Extract the name, latitude, and longitude of each named node
    df = df[['tags.name', 'lat', 'lon']].reset_index(drop=True)

    # Get only the nodes with the specific name we want, if specified
    if name:
        df = df[df['tags.name'] == name].reset_index(drop=True)

    # Make sure names are consistent for the mapping function
    df = df.rename(columns={'tags.name':'name', 'lat':'latitude', 'lon':'longitude'})
    return df

