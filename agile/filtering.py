# Functions for filtering an advertising data set
# Ernest Son
# Sam Chanow

import pandas as pd
from proximitypyhash import get_geohash_radius_approximation

def query_adid(adid, df):
    if adid == '':
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
    start_date = pd.datetime.combine(start_date, start_time)
    end_date = pd.datetime.combine(end_date, end_time)
    if start_date == '' or end_date == '':
        return
    # convert string to datetime
    df['date'] = pd.to_datetime(df['datetime'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # Parse the df by datetime
    date_filter = (df['date'] >= start_date) & (df['date'] <= end_date)
    parsed_df = df.loc[date_filter]
    print(parsed_df.head())
    parsed_df['datetime'] = parsed_df['datetime'].astype(str)
    print(parsed_df.head())
    print(len(parsed_df.index))
    return parsed_df

