from sklearn.cluster import AgglomerativeClustering
import pandas as pd

# This dataframe must contain "latitude" and "longitude"
def cluster_points(df, n):
    X = df[['latitude', 'longitude']].values
    clustering = AgglomerativeClustering(n_clusters=n)
    clustering.fit(X)
    print(clustering.labels_)

# testing
# head -1000 test_location_data_gh.csv > small_test.csv
data = pd.read_csv("../data/small_test.csv")
cluster_points(data, 10)
