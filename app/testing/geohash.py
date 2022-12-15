#Simple Script to add geohashes to test_location Dataset
#Sam Chanow

import pandas as pd
import pygeohash as gh

df = pd.read_csv(
  "../data/test_location_data.csv"
)

df["geohash"] = df.apply(lambda d : gh.encode(d.latitude, d.longitude, precision=8), axis=1)
df.to_csv("data/test_location_data_gh.csv")
