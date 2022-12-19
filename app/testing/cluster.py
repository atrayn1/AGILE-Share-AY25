# Ernest Son

# clustering will indicate high-traffic areas, still maybe potentially useful
# information later down the line

from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import pandas as pd

# This dataframe must contain "latitude" and "longitude"
def cluster_points(df, n):
    X = df[['latitude', 'longitude']].values
    clustering = AgglomerativeClustering(n_clusters=n)
    Y = clustering.fit(X)
    plt.scatter(X[:,0], X[:,1], c=Y.labels_, cmap='hot')
    #plt.show()
    #plt.savefig('/mnt/c/Users/ebs/Desktop/clusters.png')

# testing
# head -1000 test_location_data_gh.csv > small_test.csv
data = pd.read_csv("../data/one_id.csv")
size = len(data)
cluster_points(data, size - 10)
