# Beginning to work on pattern of life analysis
# Ernest Son
# Sam Chanow

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.mixture import GaussianMixture
from datetime import datetime
from math import cos, asin, sqrt, pi, atan2, sin
from os import system
import matplotlib.pyplot as plt
from random import sample

from .filtering import query_adid

pd.options.mode.chained_assignment = None

# POL Algorithm

# Preprocessing
#   Remove Anomolous Data (Too Quick)
#   Weight each Datapoint based on timestamp, sampling rate, and distance
#   Determine hyperparams for DBSCAN
# Clustering
#   Cluster with DBSCAN and return the top X clusters by summation of weight
# 2nd Clustering
#   Cluster on the top X clusters from last step and output top Y clusters
# Classification
#   Label Test Data Tampstamp -> Cluster Label

# Given two latitude and longitude points return the distance in kilometers
# TODO:
# using the numpy math functions seems to have a weird sigfig error
# investigate further...
def haversine(lat1, lon1, lat2, lon2):
    r = pi / 180
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = 0.5 * (1 - cos(dlat*r)) + cos(lat1*r) * cos(lat2*r) * 0.5 * (1 - cos(dlon*r))
    c = asin(sqrt(a))
    earth_radius = 6371
    km = 2 * earth_radius * c
    return km

# data:
#   a dataframe that contains at least ad_id, datetime, lat, long, and timestamp
#   timestamp is assumed to be a string
# speed:
#   is the cutoff speed (km/hr)
# This Dataframe must be sorted by time!!!
# Sorting is expensive... don't do it more than you have to.
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

# Cluster and return top X clusters
# Top clusters determined by summation of weight
def get_clusters(data) -> pd.DataFrame:
    # This now works with sklearn v1.2.1 (the latest version as of 2023-02-09)
    # epsilon values can be thought of as being measured in kilometers
    epsilon = 0.117
    kms_per_degree = 111.32
    threshold = epsilon / kms_per_degree
    # DBSCAN still generally works better than AgglomerativeClustering, but we
    # should keep fiddling with the AgglomerativeClustering hyperparameters
    # Results may not reflect reality either, since we're validating with a
    # limited synthetic data set
    model = DBSCAN(eps=threshold, min_samples=1, algorithm='ball_tree', metric='haversine')
    '''
    epsilon = 0.4
    kms_per_degree = 111.32
    thresh = epsilon / kms_per_degree
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=thresh)
    '''
    clusters = model.fit(data[['latitude', 'longitude']])
    labels = clusters.labels_
    return labels

# Clustering the data with GaussianMixture Model for testing purposes
def get_clusters_GM(data) -> pd.DataFrame:

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
    relevant_features = ['label', 'label_sum', 'datetime', 'travel_time', 'distance', 'speed', 'weight', 'latitude', 'longitude']
    has_max = data.label_sum == data.label_sum.max()
    max_label = data.loc[has_max, relevant_features]
    without_max = data.loc[~has_max, relevant_features]
    return max_label, without_max

def double_cluster(adid, full_data, gm=True):
    #print('Using Gaussian Mixture:' if gm else 'Using DBSCAN:')

    # Basic pre-processing
    data = query_adid(adid, full_data)
    data = data.sort_values(by='datetime')
    data = speed_filter(data, 120)
    data = weighting(data)

    # Get max weighted cluster
    labels = get_clusters_GM(data) if gm == True else get_clusters(data)
    max_group, without_max = max_cluster(data, labels)

    # Fail gracefully
    if without_max.empty:
        return None

    # Protecting GM
    if len(without_max) < 3:
        return None

    # Re-label and return new data
    new_labels = get_clusters_GM(without_max) if gm == True else get_clusters(without_max)
    without_max['label'] = new_labels
    max_new_label = without_max.label.max() + 1
    max_group['label'] = max_new_label
    full_labeled_data = pd.concat([max_group, without_max])
    return full_labeled_data

# Calculate the centroids for weighted lat/long data
# We save the cluster label groups for each centroid
# These centroids need to be saved in a Profile for prediction
def get_cluster_centroids(data) -> pd.DataFrame:
    # This should be passed something like full_labeled_data
    relevant_features = ['latitude', 'longitude', 'weight', 'label']
    label_groups = data.groupby(by='label')
    lat_long = ['latitude', 'longitude']
    def calculate_centroids(group):
        locations = group[lat_long].to_numpy()
        weights = group.weight.tolist()
        centroids = np.average(locations, axis=0, weights=weights)
        return centroids
    raw_centroids = label_groups.apply(calculate_centroids)
    centroids = raw_centroids.tolist()
    labels = raw_centroids.index
    df = pd.DataFrame(centroids, index=labels, columns=lat_long).reset_index()
    return df

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

'''
# Some sample adids to try
# 81696261-3059-7d66-69cc-67688182f974
# 54aa7153-1546-ce0d-5dc9-aa9e8e371f00
# 18665217-4566-5790-809c-702e77bdbf89
full_data = pd.read_csv('../data/weeklong.csv')

# Making some more verbose testing here
# Grabbing a random sample from the dataset and testing the predictor on it
test_adids = list(full_data.advertiser_id.unique())
#test_adids = sample(test_adids, 50)
#print("TEST IDS:", test_adids)

for adid in test_adids:
    print('For ' + adid + ':')
    # cluster, GM is default
    clustered_data = double_cluster(adid, full_data)
    n_labels = clustered_data.label.max() + 1
    model, test_accuracy = fit_predictor(clustered_data, debug=True)
'''

