import requests
import pandas as pd

# Define the Overpass API endpoint
overpass_url = "http://overpass-api.de/api/interpreter"

# Define the query to find named nodes around a given point
overpass_query = """
    [out:json];
    (
      node["name"](around:{rad}, {lat}, {lon});
    );
    out body;
"""

# Define a function to query the Overpass API for each point in a dataframe
def query_overpass(df):
    results = []
    for index, row in df.iterrows():
        rad = 100
        lat = row['lat']
        lon = row['lon']
        query = overpass_query.format(rad=rad, lat=lat, lon=lon)
        response = requests.get(overpass_url, params={'data': query})
        data = response.json()
        named_nodes = pd.json_normalize(data['elements'])
        named_nodes = named_nodes[named_nodes['tags.name'].notnull()]
        named_nodes = named_nodes[['tags.name', 'lat', 'lon']].reset_index(drop=True)
        named_nodes['query_index'] = index
        results.append(named_nodes)
    return pd.concat(results)

# Define a sample dataframe with latitudes and longitudes
df = pd.DataFrame({
    'lat': [52.5233, 52.5170, 52.5192],
    'lon': [13.4115, 13.3917, 13.4061]
})

# Query the Overpass API for named nodes around each point in the dataframe
results = query_overpass(df)

# Print the resulting dataframe
print(results)

