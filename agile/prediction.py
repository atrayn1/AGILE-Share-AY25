# Beginning to work on pattern of life analysis
# Ernest Son
# SAm

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime
from math import cos, asin, sqrt, pi

import matplotlib.pyplot as plt

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

# data:
#   a dataframe that contains at least ad_id, datetime, lat, long, and timestamp
#   timestamp is assumed to be a string
# speed:
#   is the cutoff speed (km/hr)
# This Dataframe must be sorted by time!!!
# Sorting is expensive... don't do it more than you have to.
def speed_filter(data, speed) -> pd.DataFrame:
    data['datetime'] = pd.to_datetime(data.datetime)
    data['travel_time'] = data.datetime.diff(-1) * -1 #to offset reverse diff
    # Convert timedelta to hours
    data['travel_time'] = data.travel_time.apply(lambda d : d.total_seconds() / 3600).fillna(0)
    # Get distance
    data['next_latitude'] = data.latitude.shift(-1)
    data['next_longitude'] = data.longitude.shift(-1)

    # Given two latitude and longitude points return the distance in kilometers
    # TODO:
    # using the numpy math functions seems to have a weird sigfig error
    # investigate further...
    def haversine(lat1, lon1, lat2, lon2):
        p = pi / 180
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = 0.5 - cos(dlat*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos(dlon*p))/2
        c = asin(sqrt(a))
        earth_radius = 6371
        km = 2 * earth_radius * c
        return km

    # Throw the dataframe into an apply that calculates distance
    data['distance'] = data.apply(lambda row: haversine(row.latitude, row.longitude, 
                                              row.next_latitude, row.next_longitude), axis=1)
    # Calculate speed
    data['speed'] = data.distance / data.travel_time

    # drop bad rows
    filtered_data = data.loc[data.speed < speed]

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
    # gamma
    threshold = 0.05
    mask = data.distance <= threshold
    data['weights'] = data.weights.where(mask, other=1)
    return data

# Cluster and return top X clusters
# Top clusters determined by summation of weights
def get_clusters(data, debug=False) -> pd.DataFrame:

    # Returns optimal hyperparameters for DBSCAN clustering algorithm
    # We use reasonable defaults right now
    def optimize(data):
        # p (rho) is a hyperparameter, values can be 0.1, 0.25, or 0.3
        # epsilon is another hyperparameter, can be 0.2km or 0.3km
        epsilon = 0.2
        p = 0.1
        kms_per_degree = 111
        dist = epsilon / kms_per_degree
        min_samples = int(data.weights.sum() * p)
        return epsilon, min_samples

    eps, min_samples = optimize(data)
    model = DBSCAN(eps=eps,
            min_samples=min_samples,
            algorithm='ball_tree',
            metric='haversine')
    db = model.fit(data[['latitude', 'longitude']], sample_weight=data.weights)
    labels = db.labels_

    if debug:
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
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
                xy.latitude,
                xy.longitude,
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy.latitude,
                xy.longitude,
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.savefig('clusters.png')

# Train
def pol_train(train_data):
    pass

# Test and predict
def pol_predict(test_data) -> pd.DataFrame:
    pass

# Test accuracy of predict
def pol_accuracy(prediction_data, gold_labels):
    pass

#Test for Lat and Long coordinates
#lat1 = 38.978015
#lon1 = -76.504426

#lat2 = 38.981220
#lon2 = -76.487589

data = pd.read_csv("../data/test.csv")
data.sort_values(by=['datetime'], inplace=True)
data = speed_filter(data, 120)
data = weighting(data)
print(data[['datetime', 'travel_time', 'distance', 'speed', 'weights']])
get_clusters(data, debug=True)

