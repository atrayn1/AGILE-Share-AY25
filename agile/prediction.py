# Beginning to work on pattern of life analysis
# Ernest Son
# Sam Chanow

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
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

    # For easier graphing use later
    data['X'] = data.latitude
    data['Y'] = data.longitude

    # Normalize the weights from 0 to 1
    data['weights'] = (data.weights - data.weights.min()) / (data.weights.max() - data.weights.min())
    return data

# Cluster and return top X clusters
# Top clusters determined by summation of weights
def get_clusters(data, debug=False) -> pd.DataFrame:

    # Returns optimal hyperparameters for DBSCAN clustering algorithm
    # We use reasonable defaults right now
    def optimize(data):
        # rho is a hyperparameter, values can be 0.1, 0.25, or 0.3
        # epsilon is another hyperparameter, can be 0.2km or 0.3km
        epsilon = 0.1
        rho = 0.1
        kms_per_degree = 111
        eps = epsilon / kms_per_degree
        min_samples = int(data.weights.sum() * rho)
        return eps, min_samples

    eps, min_samples = optimize(data)
    model = DBSCAN(eps=eps,
            min_samples=min_samples,
            algorithm='ball_tree',
            metric='haversine')
    db = model.fit(data[['X', 'Y']], sample_weight=data.weights)
    labels = db.labels_

    if debug:
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        # Visualize clusters on a map
        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = data[class_member_mask & core_samples_mask]
            plt.plot(
                xy.X,
                xy.Y,
                'o',
                markerfacecolor=tuple(col),
                markeredgecolor='k',
                markersize=14,
            )

            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy.X,
                xy.Y,
                'o',
                markerfacecolor=tuple(col),
                markeredgecolor='k',
                markersize=6,
            )

        plt.title(f'Estimated number of clusters: {n_clusters_}')
        plt.savefig('clusters.png')
    return labels

def max_cluster(data, labels) -> int:
    data['label'] = labels
    data['label_sum'] = data.groupby('label').weights.transform('sum')
    max_label = data.loc[data.label_sum == data.label_sum.max()]
    without_max = data.loc[data.label_sum != data.label_sum.max()]
    #max_label = data.loc[data.label_sum == data.label_sum.max(), ['label', 'datetime', 'travel_time', 'distance', 'speed', 'weights', 'X', 'Y']]
    #without_max = data.loc[data.label_sum != data.label_sum.max(), ['label', 'datetime', 'travel_time', 'distance', 'speed', 'weights', 'X', 'Y']]
    return max_label, without_max

def double_cluster(adid, full_data):
    # Basic pre-processing
    data = query_adid(adid, full_data)
    data = data.sort_values(by='datetime')
    data = speed_filter(data, 120)
    data = weighting(data)
    labels = get_clusters(data)
    max_group, without_max = max_cluster(data, labels)
    # Fail gracefully
    if without_max.empty:
        return None
    # Re-label and return new data
    new_labels = get_clusters(without_max)
    without_max['label'] = new_labels
    max_new_label = without_max.label.max() + 1
    max_group['label'] = max_new_label
    full_labeled_data = pd.concat([max_group, without_max])
    return full_labeled_data

# Cluster centroid calculations
# These centroids need to be saved in a Profile for prediction
# TODO we need to figure out how to use the weights when sent into the NearestCentroid classifier
# TODO we need to make sure that the label coming out of this function is the same as the label 
# being used for the classifier (Otherwise we are dead in the water)
def get_cluster_centroids(data) -> pd.DataFrame:
    # This should be passed something like full_labeled_data
    X = data[['latitude', 'longitude']]
    y = data.label
    clf = NearestCentroid()
    clf.fit(X, y)
    centroids = pd.DataFrame(clf.centroids_, columns=['latitude', 'longitude'])
    # Adding a label column for clarity
    centroids['label'] = clf.classes_ 

    return centroids

def get_top_N_clusters(data, N) -> pd.DataFrame:
    ordered_labels = data.sort_values(by='label_sum', ascending=False).drop_duplicates(subset=['label']).head(N).label
    top_N_cluster_data = data[data.label.isin(ordered_labels)]

    return get_cluster_centroids(top_N_cluster_data)

    #X = top_N_cluster_data[['latitude', 'longitude']]
    #y = top_N_cluster_data.label
    #clf = NearestCentroid()
    #clf.fit(X, y)
    #centroids = pd.DataFrame(clf.centroids_, columns=['latitude', 'longitude'])
    #return centroids

# Full model pipeline
def fit_predictor(clustered_data, debug=False) -> RandomForestClassifier:
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
    X_train, X_test, y_train, y_test = train_test_split(classifier_data.drop('label', axis=1), classifier_data.label, test_size=0.25)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    if debug:
        #system('clear')
        print('With', clustered_data.label.max() + 1, 'labels, Accuracy:', model.score(X_test, y_test))
    # This function should return the actual trained model for later use as well as a score to see how it did
    return model, model.score(X_test, y_test)

'''
full_data = pd.read_csv('../data/weeklong.csv')
# Some sample adids to try
# 81696261-3059-7d66-69cc-67688182f974
# 54aa7153-1546-ce0d-5dc9-aa9e8e371f00
# 18665217-4566-5790-809c-702e77bdbf89
adid = '54aa7153-1546-ce0d-5dc9-aa9e8e371f00'
clustered_data = double_cluster(adid, full_data)
lois = get_top_N_clusters(clustered_data, 4)

print(lois.head())

centroids = get_cluster_centroids(clustered_data)
#print(centroids.head())
model, test_accuracy = fit_predictor(clustered_data, debug=True)
accuracy = 0.0
for adid in tqdm(full_data['advertiser_id'].unique()):
    clustered_data = double_cluster(adid, full_data)
    model, test_accuracy = fit_predictor(clustered_data, debug=True)
    accuracy += test_accuracy
mean_accuracy = accuracy / len(full_data.advertiser_id.unique())
print('Average Accuracy:', mean_accuracy)
'''

