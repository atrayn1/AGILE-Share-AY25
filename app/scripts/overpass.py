#SAMUEL CHANOW
#TESTING OVERPASS API

#Length in km of 1° of latitude = always 111.32 km
#Length in km of 1° of longitude = 40075 km * cos( latitude ) / 360

#node(around:150, 51.50069, -0.12458)[natural=tree];

import overpy
import pandas as pd

api = overpy.Overpass()

df = pd.read_csv(
  "../data/test_location_data_gh.csv"
)
query = "node(around:1000, " + str(df['latitude'][0]) + ", " + str(df['longitude'][0]) + "); out body;"

print(query)

# fetch all ways and nodes
result = api.query(query)

print(result)

print(len(result.ways))
print(len(result.nodes))
print(len(result.relations))

#All of the named Nodes could be useful
#We could grab all named Nodes and use that to pick out info that we need
for node in result.nodes:
    print("Name: %s" % node.tags.get("name", "n/a"))

print("SUCCESS")

