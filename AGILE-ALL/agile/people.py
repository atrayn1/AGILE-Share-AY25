import pandas as pd
from datetime import datetime, timedelta

def colocation(data: pd.DataFrame, lois: pd.DataFrame, duration: int) -> pd.DataFrame:
    """
    Finds devices that were detected within a certain distance and time range of a given set of locations of interest.

    Args:
        data: A DataFrame containing all of the data.
        lois: A DataFrame of locations of interest for a single advertiser ID.
        duration: An int representing the time range in hours to match timeframes of datapoints.

    Returns:
        A DataFrame containing co-located devices.
    """
    relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']
    data = data[relevant_features].sort_values(by="datetime").reset_index(drop=True)

    try:
        filtered = data[data.geohash.isin(lois.geohash.unique())]
    except:
        return pd.DataFrame()
    
    try:
        filtered = filtered[filtered.advertiser_id != lois.advertiser_id.iloc[0]]
    except:
        pass
    
    print(filtered)
    
    search_time = timedelta(hours=duration)
    
    def time_filter(row):
        loi_filtered = lois[lois.geohash == row.geohash]
        loi_dates = pd.to_datetime(loi_filtered.datetime)
        filtered_time = pd.to_datetime(row.datetime)
        within_timerange = (filtered_time > (loi_dates - search_time)) & (filtered_time < (loi_dates + search_time))
        row['remove'] = not within_timerange.any()
        return row

    filtered = filtered.apply(time_filter, axis=1)
    data_out = None
    if 'remove' in filtered.columns:
        data_out = filtered.loc[filtered.remove == False].drop(columns=['remove'])

    return data_out

