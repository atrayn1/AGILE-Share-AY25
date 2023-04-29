import pandas as pd
from proximitypyhash import get_geohash_radius_approximation
from datetime import datetime as dt
import requests

def query_adid(adid: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe by advertiser_id and return a new dataframe.
    """
    if not adid or df.empty:
        return None
    parsed_df = df.loc[df.advertiser_id == adid]
    return parsed_df

def query_location(lat: str, long: str, radius: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe by latitude, longitude, and radius, and return a new dataframe.
    MUST HAVE GEOHASH PRECISION 10 or this will not work
    """
    if not lat or not long or not radius or df.empty:
        return None
    lat = float(lat)
    long = float(long)
    radius = float(radius)

    # Get geohashes in the given radius at the given point
    geohashes = get_geohash_radius_approximation(
        latitude=lat,
        longitude=long,
        radius=radius,
        precision=10,
        georaptor_flag=False,
        minlevel=1,
        maxlevel=12)

    # Create a new smaller dataframe
    parsed_df = df.loc[df.geohash.isin(geohashes)]
    return parsed_df

def query_date(start_date: str, start_time: str, end_date: str, end_time: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe by date range and return a new dataframe.
    """
    if not start_date or not end_date or df.empty:
        return None

    # Combine dates and times from user input
    start = dt.combine(pd.to_datetime(start_date).date(), start_time)
    end = dt.combine(pd.to_datetime(end_date).date(), end_time)

    # Parse the df by datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    date_filter = (df['datetime'] >= start) & (df['datetime'] <= end)
    parsed_df = df.loc[date_filter].copy()
    return parsed_df

def query_node(lat: float, lon: float, rad: float, name: str = '') -> pd.DataFrame:
    """
    Find named nodes within a specified radius of a given latitude and longitude.

    Parameters:
        lat (float): latitude of the center point
        lon (float): longitude of the center point
        rad (float): radius in meters to search around the center point
        name (str, optional): name of the node to filter by, default is '' which returns all named nodes

    Returns:
        pd.DataFrame: a DataFrame with columns 'name', 'latitude', and 'longitude' for each named node within the specified radius and (optionally) with the given name
    """

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

    # Fail gracefully when no nodes are found, return None object
    if df.empty:
        return None

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

