# Ernest Son
from geopy.geocoders import Nominatim
from geopy.point import Point
import pandas as pd
import numpy as np

# need to specify a user agent or the api bitches at you
geolocator = Nominatim(user_agent='usna')

def reverse_geocoding(lat, lon):
    try:
        location = geolocator.reverse(Point(lat, lon))
        print(location.raw['display_name'])
        return location.raw['display_name']
    except:
        return None

# This dataframe must contain 'latitude' 'longitude' 'datetime'
def reverse_geocode(df):
    df.loc[:, ('dates')] = pd.to_datetime(df['datetime']) # No setting with copy error
    df.sort_values(by="dates")
    df['address'] = np.vectorize(reverse_geocoding)(df['latitude'], df['longitude'])
    return df 

# testing
'''
# head -1000 test_location_data_gh.csv > small_test.csv
data = pd.read_csv("../data/small_test.csv")
test = reverse_geocode(data)
print(test['address'].unique())
'''
