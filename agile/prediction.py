import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.mixture import GaussianMixture
from datetime import datetime
from math import cos, asin, sqrt, pi, atan2, sin
from os import system
from random import sample
from pygeohash import encode

from .filtering import query_adid

pd.options.mode.chained_assignment = None

# Given two latitude and longitude points return the distance in kilometers
def haversine(lat1, lon1, lat2, lon2):
    r = pi / 180
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = 0.5 * (1 - cos(dlat*r)) + cos(lat1*r) * cos(lat2*r) * 0.5 * (1 - cos(dlon*r))
    c = asin(sqrt(a))
    earth_radius = 6371
    km = 2 * earth_radius * c
    return km

def speed_filter(data, speed) -> pd.DataFrame:
    filtered_data = data.copy(deep=True)
    filtered_data['datetime'] = pd.to_datetime(data.datetime)
    filtered_data['travel_time'] = filtered_data.datetime.diff(-1) * -1 #to offset reverse diff
    # Convert timedelta to hours
    filtered_data['travel_time'] = filtered_data.travel_time.apply(lambda d : d.total_seconds() / 3600).fillna(0)
    # Get distance
    filtered_data['next_latitude'] = filtered_data.latitude.shift(-1)
    filtered_data['next_longitude'] = filtered_data.longitude.shift(-1)

    # Throw the dataframe into an apply that calculates distance
    filtered_data['distance'] = filtered_data.apply(lambda row: haversine(row.latitude, row.longitude, row.next_latitude, row.next_longitude), axis=1)
    # Calculate speed
    filtered_data['speed'] = filtered_data.distance / filtered_data.travel_time

    # drop bad rows
    filtered_data = filtered_data[filtered_data.speed < speed]

    return filtered_data

# Creates 'weight' feature in input dataframes
# The Dataframe must be sorted by time!!!
def weighting(data) -> pd.DataFrame:
    # Give every row a weight equal to the number of seconds before the next row
    data['datetime'] = pd.to_datetime(data.datetime)
    data['travel_time'] = data.datetime.diff(-1) * -1

    # Convert timedelta to seconds and impute last row with 0 (no more data)
    data['travel_time'] = data.travel_time.apply(lambda d : d.total_seconds()).fillna(0)

    # Find sampling rate of data
    sampling_rate = data.travel_time.median()

    # Calculate the weight depending on distances between datapoints
    data['weight'] = data.travel_time / sampling_rate
    # threshold is in km
    threshold = 0.05
    mask = data.distance <= threshold
    data['weight'] = data.weight.where(mask, other=1)

    # Normalize the weight from 0 to 1
    #data['weight'] = (data.weight - data.weight.min()) / (data.weight.max() - data.weight.min())
    return data

# Clustering the data with GaussianMixture Model for testing purposes
def get_clusters(data) -> pd.DataFrame:

    def since_midnight(now) -> int:
        return (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    data['seconds'] = data.datetime.apply(since_midnight)

    model = GaussianMixture(n_components=300 if data.shape[0] >= 300 else data.shape[0], 
                            covariance_type='full',
                            max_iter=200,
                            init_params='random',
                            verbose=0)
    clusters = model.fit(data[['latitude', 'longitude', 'seconds']])
    labels = model.predict(data[['latitude', 'longitude', 'seconds']])

    return labels

def max_cluster(data, labels) -> int:
    data['label'] = labels
    data['label_sum'] = data.groupby('label').weight.transform('sum')
    relevant_features = ['geohash', 'advertiser_id', 'label', 'label_sum', 'datetime', 'travel_time', 'distance', 'speed', 'weight', 'latitude', 'longitude']
    has_max = data.label_sum == data.label_sum.max()
    max_label = data.loc[has_max, relevant_features]
    without_max = data.loc[~has_max, relevant_features]
    return max_label, without_max

def double_cluster(adid, full_data):
    # Basic pre-processing
    data = query_adid(adid, full_data)
    data = data.sort_values(by='datetime')
    data = speed_filter(data, 120)
    data = weighting(data)

    # Get max weighted cluster
    labels = get_clusters(data)
    max_group, without_max = max_cluster(data, labels)

    # Fail gracefully
    if without_max.empty:
        return None

    # Protecting GM
    if len(without_max) < 3:
        return None

    # Re-label and return new data
    new_labels = get_clusters(without_max)
    without_max['label'] = new_labels
    max_new_label = without_max.label.max() + 1
    max_group['label'] = max_new_label
    full_labeled_data = pd.concat([max_group, without_max])
    return full_labeled_data

def get_cluster_centroids(data) -> pd.DataFrame:

    # Ensure timestamp column is explicitly of type 'datetime'
    data['datetime'] = pd.to_datetime(data.datetime)

    # Calculate midpoint timestamps for each cluster
    midpoints = data.groupby('label').agg({'datetime': ['min', 'max']})
    midpoints['midpoint'] = midpoints.apply(lambda row: row['datetime']['min'] +
                                             (row['datetime']['max'] - row['datetime']['min']) / 2, axis=1)
    midpoints = midpoints['midpoint'].to_dict()

    # Calculate weighted centroids for each cluster
    centroids = data.groupby('label').apply(lambda x: np.average(x[['latitude', 'longitude']],
                                                                 weights=x['weight'], axis=0))
    centroids = pd.DataFrame(centroids.tolist(), columns=['latitude', 'longitude'], index=centroids.index)

    # Add midpoint timestamp column to centroids dataframe
    centroids['datetime'] = centroids.index.map(lambda x: midpoints[x])

    # Re-geohash the lats and longs, since we have a new lat-long pair
    centroids['geohash'] = centroids.apply(lambda x: encode(x.latitude, x.longitude, precision=10), axis=1)

    # Label the centroids with the original adid
    centroids['advertiser_id'] = data.advertiser_id.iloc[0]

    return centroids


def get_top_N_clusters(data, N) -> pd.DataFrame:
    if data is None:
        return None
    ordered_data = data.sort_values(by='label_sum', ascending=False)
    ordered_labels = ordered_data.drop_duplicates(subset='label').head(N).label
    top_N_cluster_data = data[data.label.isin(ordered_labels)]
    return get_cluster_centroids(top_N_cluster_data)

# Full model pipeline
def fit_predictor(clustered_data, debug=False) -> RandomForestClassifier:
    if clustered_data is None:
        # No model, zero accuracy
        return None, 0
    classifier_data = pd.DataFrame()
    # Return number of seconds since midnight
    def since_midnight(now) -> int:
        return (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    classifier_data['seconds'] = clustered_data.datetime.apply(since_midnight)
    classifier_data['dayofweek'] = clustered_data.datetime.dt.dayofweek
    classifier_data['label'] = clustered_data.label
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(classifier_data.drop('label', axis=1), classifier_data.label, test_size=0.25)
    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    n_labels = clustered_data.label.max() + 1
    model_accuracy = model.score(X_test, y_test)
    if debug:
        rounded_accuracy = round(100 * model_accuracy, 2)
        print('With', n_labels, 'labels, Accuracy:', rounded_accuracy)
        # outlier detection
        testing_data = X_test
        testing_data['pred'] = model.predict(X_test)
        testing_data['real'] = y_test
        testing_data['outlier'] = testing_data.real != testing_data.pred
        n_outliers = testing_data.outlier.sum()
        outlier_percent = round(100 * n_outliers / len(testing_data), 2)
        ops = '(' + str(outlier_percent) + '%)'
        print(n_outliers, 'possible outliers detected in a set of', len(testing_data), 'test data points', ops)
        print()

    # This function should return the actual trained model for later use as well as a score to see how it did
    return model, model_accuracy

