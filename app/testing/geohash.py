#Simple Script to add geohashes to test_location Dataset
#Sam Chanow

import pandas as pd
import pygeohash as gh

df = pd.read_csv(
  "../data/_54aa7153-1546-ce0d-5dc9-aa9e8e371f00_weeklong.csv"
)

df["geohash"] = df.apply(lambda d : gh.encode(d.latitude, d.longitude, precision=4), axis=1)
df.to_csv("../data/_54aa7153-1546-ce0d-5dc9-aa9e8e371f00_weeklong_gh.csv")
