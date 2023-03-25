import pandas as pd
import requests
import xml.etree.ElementTree as ET

# Queries Overpass to find nodes within a given radius for a list of coordinates
def find_all_nearby_nodes(data, radius):

    # Create a list of location pairs as a list of strings
    points = [f"{lat},{lon}" for lat, lon in zip(data['latitude'], data['longitude'])]

    # Join the list of location pairs into a single string
    polyline_str = ",".join(points)

    # Define query to look for named nodes within the specified radius
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"node(around:{radius},{polyline_str});out body;"

    # Send the query to the Overpass API
    # Must be a POST request, typically too large to be a GET request
    response = requests.post(overpass_url, data=query)

    # Raises exception when not a 2xx response
    response.raise_for_status()

    # Parse the XML response using ElementTree
    root = ET.fromstring(response.content)

    # Extract named nodes and their corresponding latitude and longitude
    nodes = []
    for node in root.findall('.//node'):
        name = node.find('tag[@k="name"]')
        if name is not None:
            lat = node.get('lat')
            lon = node.get('lon')
            nodes.append({'name': name.get('v'), 'lat': lat, 'lon': lon})

    # Convert the list of nodes to a dataframe
    df = pd.DataFrame(nodes)

    return df

