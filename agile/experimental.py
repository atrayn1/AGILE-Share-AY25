# Beginning to work on pattern of life analysis
# Ernest Son
# Sam Chanow

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from math import cos, asin, sqrt, pi, atan2, sin
from filtering import query_adid
from tqdm import tqdm
from os import system
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

# POL Algorithm

# Preprocessing
#   Remove Anomolous Data (Too Quick)
#   Weight each Datapoint based on timestamp, sampling rate, and distance
#   Determine hyperparams for DBSCAN
# Clustering
#   Cluster with DBSCAN and return the top X clusters by summation of weights
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

    # For easier graphing use later
    filtered_data['X'] = filtered_data.latitude
    filtered_data['Y'] = filtered_data.longitude

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

    # Calculate the weights depending on distances between datapoints
    data['weights'] = data.travel_time / sampling_rate
    # threshold is in km
    threshold = 0.05
    mask = data.distance <= threshold
    data['weights'] = data.weights.where(mask, other=1)

    # Normalize the weights from 0 to 1
    data['weights'] = (data.weights - data.weights.min()) / (data.weights.max() - data.weights.min())
    return data

# Identify clusters of datapoints limited to be within a specific radius
def get_clusters(data, debug=False) -> pd.DataFrame:
    kms_per_degree = 111.32
    #epsilon = 0.5
    epsilon = 0.7
    thresh = epsilon / kms_per_degree
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=thresh, linkage='complete', metric='l1')
    clusters = model.fit(data[['X', 'Y']])
    return clusters.labels_

# Return top X clusters
# Top clusters determined by summation of weights
def max_cluster(data, labels) -> int:
    data['label'] = labels
    data['label_sum'] = data.groupby('label').weights.transform('sum')
    relevant_features = ['label', 'datetime', 'travel_time', 'distance', 'speed', 'weights', 'X', 'Y']
    home_mask = data.label_sum == data.label_sum.max()
    max_label = data.loc[home_mask, relevant_features]
    without_max = data.loc[~home_mask, relevant_features]
    return max_label, without_max

def double_cluster(adid, full_data):
    # Get the initial clusters
    data = query_adid(adid, full_data)
    data = data.sort_values(by='datetime')
    data = speed_filter(data, 120)
    labels = get_clusters(data)

    # Weight the clusters to identify the max cluster
    data = weighting(data)
    max_group, without_max = max_cluster(data, labels)

    # Fail gracefully
    if without_max.empty:
        return None

    # Get clusters on the data with the max cluster removed
    new_labels = get_clusters(without_max)
    without_max['label'] = new_labels

    # Re-add the max cluster and return overall data
    max_new_label = without_max.label.max() + 1
    max_group['label'] = max_new_label
    full_labeled_data = pd.concat([max_group, without_max])
    return full_labeled_data

# Full model pipeline
def fit_predictor(clustered_data, debug=False):
    if clustered_data is None:
        # No model, zero accuracy
        return None, 0
    # Make data usable for classifier
    # Input is time (seconds since 0000)
    # Label is label
    classifier_data = pd.DataFrame()

    # Return number of seconds since midnight
    def since_midnight(now) -> int:
        return (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

    classifier_data['seconds'] = clustered_data.datetime.apply(since_midnight)
    classifier_data['label'] = clustered_data.label

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(classifier_data.drop('label', axis=1), classifier_data.label, test_size=0.1)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    if debug:
        #system('clear')
        num_labels = clustered_data.label.max() + 1
        accuracy = model.score(X_test, y_test)
        print('With', num_labels, 'labels, Accuracy:', accuracy)
    # This function should return the actual trained model for later use as well
    # as a score to see how it did
    return model, model.score(X_test, y_test)

full_data = pd.read_csv('../data/weeklong.csv')
# Some sample adids to try
# 81696261-3059-7d66-69cc-67688182f974
'''
adid = "54aa7153-1546-ce0d-5dc9-aa9e8e371f00"
# 18665217-4566-5790-809c-702e77bdbf89
clustered_data = double_cluster(adid, full_data)
model, test_accuracy = fit_predictor(clustered_data, debug=True)
'''
accuracy = 0.0
#for adid in tqdm(full_data['advertiser_id'].unique()):
for adid in full_data['advertiser_id'].unique():
    clustered_data = double_cluster(adid, full_data)
    model, test_accuracy = fit_predictor(clustered_data, debug=True)
    accuracy += test_accuracy
mean_accuracy = accuracy / len(full_data.advertiser_id.unique())
print('Average Accuracy:', mean_accuracy)

