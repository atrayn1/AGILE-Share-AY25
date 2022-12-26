# Ernest Son
# testing reverse geohashing using Nominatim (OpenStreetMap)
# pip3 install geopy
from geopy.geocoders import Nominatim

# need to specify a user agent or OSM bitches at you
geolocator = Nominatim(user_agent='not_the_government@nsa.gov')
location = geolocator.reverse("38.99611979598881, -77.03012409495409")
print(location.address)
print((location.latitude, location.longitude))
