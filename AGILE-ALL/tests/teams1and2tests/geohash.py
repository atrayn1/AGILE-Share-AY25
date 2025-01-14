# Simple Script to add geohashes to test_location Dataset
# Sam Chanow

import pandas as pd
import pygeohash as gh

df = pd.read_csv(
  '../data/_07e28697-2930-a575-6c51-2267182152f8_weeklong.csv'
#  '../data/weeklong.csv'
)

df['geohash'] = df.apply(lambda d : gh.encode(d.latitude, d.longitude, precision=10), axis=1)
relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id', 'horizontal_accuracy', 'carrier', 'model', 'wifi_ssid', 'wifi_bssid']
df = df[relevant_features]
df.to_csv('../data/_07e28697-2930-a575-6c51-2267182152f8_weeklong_gh.csv')
#df.to_csv('../data/weeklong_gh.csv')

