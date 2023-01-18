# Beginning to work on pattern of life analysis
# Ernest Son

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime

def pattern_of_life(data, adid, debug=False) -> pd.DataFrame:

    # These are the features we care about in the input dataframe
    relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']
    data = data[relevant_features]

    # This is the output dataframe, i.e. where we store suspicious geocodes
    data_out = pd.DataFrame(columns=relevant_features)

    # Sort so we only have the adids we care about
    data = pd.DataFrame(data[data.advertiser_id == adid])

    # Sort by time and convert to proper datetime objects
    data['datetime'] = pd.to_datetime(data.datetime)
    data.sort_values(by='datetime', ascending=True, inplace=True)

    # Find sampling rate of data
    data['intervals'] = data.datetime.diff()
    data['intervals'] = data.intervals.fillna(pd.Timedelta(seconds=0))
    sampling_rate = data.intervals.median()

    print(data[['latitude', 'longitude']])
    # vectorized haversine function
    def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
        '''
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees or in radians)
        All lat/lon coordinates must have numeric dtypes and be of equal length.
        '''
        if to_radians:
            lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
        a = np.sin((lat2-lat1)/2.0)**2 + \
            np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
        return earth_radius * 2 * np.arcsin(np.sqrt(a))

    lat1 = data.latitude
    lat2 = data.latitude.shift()
    lon1 = data.longitude
    lon2 = data.longitude.shift()
    data['distances'] = haversine(lat1, lat2, lon1, lon2)

    data['weights'] = (data.intervals / sampling_rate)
    threshold = 0.05
    data['weights'] = data['weights'].loc[data.distances > threshold ] = 1

    print(data[['distances', 'latitude', 'longitude']])

    # p is a hyperparameter, values can be 0.1, 0.25, or 0.3
    # epsilon is another hyperparameter, can be 0.2km or 0.3km
    epsilon = 0.2
    p = 0.1
    kms_per_degree = 111
    dist = epsilon / kms_per_degree
    min_samples = int(data.weights.sum() * p)
    model = DBSCAN(eps=dist,
            min_samples=min_samples,
            algorithm='ball_tree',
            metric='haversine')
    X = model.fit(data[['latitude', 'longitude']], sample_weight=data.weights)

data = pd.read_csv('AGILE/data/weeklong_gh.csv')
ubl = "54aa7153-1546-ce0d-5dc9-aa9e8e371f00"
pattern_of_life(data, ubl, debug=True)

