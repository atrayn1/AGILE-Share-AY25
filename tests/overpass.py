# SAMUEL CHANOW
# TESTING OVERPASS API

# Length in km of 1° of latitude = always 111.32 km
# Length in km of 1° of longitude = 40075 km * cos( latitude ) / 360

# node(around:150, 51.50069, -0.12458)[natural=tree];

import overpy
import pandas as pd

api = overpy.Overpass()

df = pd.read_csv(
  "../data/weeklong.csv"
)
def get_node(row):
    range = 25
    query = "node(around:" + str(range) + ", " + str(row.latitude) + ", " + str(row.longitude) + "); out body;"
    print(query)
    '''
    print('found', len(result.ways), 'ways')
    print('found', len(result.nodes), 'nodes')
    print('found', len(result.relations), 'relations')
    '''
    result = api.query(query)
    for node in result.nodes:
        name = node.tags.get('name')
        print(name)
df.apply(get_node, axis=1)
print(data_out)

