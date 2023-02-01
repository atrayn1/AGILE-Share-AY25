# Ernest Son
from geopy.geocoders import Nominatim
from geopy.point import Point
import pandas as pd
import numpy as np
from requests import head, ConnectionError

# need to specify a user agent or the api bitches at you
geolocator = Nominatim(user_agent='usna')

def reverse_geocoding(lat, lon):
    try:
        api_status_url = 'https://nominatim.openstreetmap.org/status.php'
        r = head(api_status_url)
        if r.status_code != 200:
            print('api down')
            return None
        location = geolocator.reverse(Point(lat, lon))
        return location.raw['display_name']
    except ConnectionError:
        print('failed to connect')
        return None

# This dataframe must contain 'latitude' 'longitude' 'datetime'
def reverse_geocode(df):
    # Fail gracefully if nothing is provided
    # Make the address column anyway
    if df.empty:
        df['address'] = pd.Series(dtype='string')
        return df
    df['address'] = np.vectorize(reverse_geocoding)(df.latitude, df.longitude)
    return df

# testing
'''
# head -1000 test_location_data_gh.csv > small_test.csv
data = pd.read_csv("../data/small_test.csv")
test = reverse_geocode(data)
print(test['address'].unique())
'''
