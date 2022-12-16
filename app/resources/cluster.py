from sklearn.cluster import AgglomerativeClustering
import pandas as pd

# need to specify a user agent or the api bitches at you

# This dataframe must contain 'latitude' 'longitude' 'datetime'
def cluster_points(df, n):
    X = df[['latitude', 'longitude']].values
    clustering = AgglomerativeClustering(n_clusters=n)
    clustering.fit(X)
    df['address'] = np.vectorize(reverse_geocoding)(df['latitude'], df['longitude'])
    print('reverse_geocode complete!')
    return df 

# testing
# head -1000 test_location_data_gh.csv > small_test.csv
data = pd.read_csv("../data/small_test.csv")
test = cluster_points(data)
