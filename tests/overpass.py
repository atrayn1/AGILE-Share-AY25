# Playing around with overpass

# Length in km of 1° of latitude = always 111.32 km
# Length in km of 1° of longitude = 40075 km * cos( latitude ) / 360

# example query:
# node(around:150, 51.50069, -0.12458)[natural=tree];

# There is some clunkiness involved with querying exact node names, but it's
# a start.

import pandas as pd
import requests

#import overpy
#api = overpy.Overpass()

df = pd.read_csv(
  "../data/test.csv"
)
def get_node(row):
    range = 1000
    #query = "node(around:" + str(range) + ", " + str(row.latitude) + ", " + str(row.longitude) + "); out body;"
    print(query)
    result = api.query(query)
    for node in result.nodes:
        name = node.tags.get('name')
        if name is not None:
            print(name)
df.apply(get_node, axis=1)
print(data_out)

