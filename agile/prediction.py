# Beginning to work on pattern of life analysis
# Ernest Son
# SAm

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime

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

# Data is a dataframe that contains at least ad_id, timestamp, lat, long
#TODO SAM
def speed_filter(data) -> pd.DataFrame:
    pass

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
