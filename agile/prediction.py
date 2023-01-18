# Beginning to work on pattern of life analysis
# Ernest Son
# SAm

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime
from math import cos, asin, sqrt, pi

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
def geo_distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) #2*R*asin...

# Data is a dataframe that contains at least ad_id, datetime, lat, long, and timestamp needs to be string
# Speed is the cutoff speed (km/hr)
#TODO SAM
def speed_filter(data, speed) -> pd.DataFrame:
    data['date'] = pd.to_datetime(data['datetime'])
    data['timediff'] = data['date'].diff()
    #Convert timedelta to hours
    data['timediff'] = data['timediff'].apply(lambda d : d.total_seconds() / 3600)
    #Get distance
    data['next_latitude'] = data['latitude'].shift(1)
    data['next_longitude'] = data['longitude'].shift(1)
    #Throw the dataframe into an apply that passes everyhting needed to geo_distance
    data['distance'] = data.apply(lambda row : geo_distance(row.latitude, row.longitude, 
                                                row.next_latitude, row.next_longitude), axis=1)
    #Calculate speed
    data['speed'] = data['distance'] / data['timediff']

    #drop bad rows
    filtered_data = df.loc[df['speed'] < speed]

    return filtered_data

# Creates 'weight' feature in input dataframes
#TODO SAM
def weighting(data) -> pd.DataFrame:
    pass

# Returns optimal hyperparameters for DBSCAN clustering algorithm
#TODO ERNIE
def optimize(data):
    pass

# TODO stop here

# Cluster and return top X clusters
# Top clusters determined by summation of weights
def dbscan_cluster(data, X) -> pd.DataFrame:
    pass

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

#print(geo_distance(lat1, lon1, lat2, lon2))
df = pd.read_csv("../data/demo_2023-01-11.csv")
df = speed_filter(df, 120)
df.to_csv('sample_time_filter.csv')

print(df.head())