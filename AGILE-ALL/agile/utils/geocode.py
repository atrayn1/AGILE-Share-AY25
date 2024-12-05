from geopy.geocoders import Nominatim
from geopy.point import Point
import pandas as pd
import numpy as np
from requests import head, ConnectionError

# Create a geolocator object with a user agent
geolocator = Nominatim(user_agent='usna')

def reverse_geocoding(lat: float, lon: float) -> str:
    try:
        # Check if the API is up and running
        if head('https://nominatim.openstreetmap.org/status.php').status_code != 200:
            print('API down')
            return None
        # Reverse geocode the given latitude and longitude
        location = geolocator.reverse(Point(lat, lon))
        return location.raw['display_name']
    except ConnectionError:
        print('Failed to connect')
        return None

# This function takes a DataFrame containing columns 'latitude', 'longitude', and 'datetime'
# and adds a new column 'address' containing the reverse geocoded address for each row
def reverse_geocode(df: pd.DataFrame) -> pd.DataFrame:
    # If the DataFrame is empty, just add an empty 'address' column
    if df.empty:
        df['address'] = pd.Series(dtype='string')
        return df
    # Otherwise, use vectorization to apply the reverse_geocoding function to each row
    df['address'] = np.vectorize(reverse_geocoding)(df.latitude, df.longitude)
    return df

