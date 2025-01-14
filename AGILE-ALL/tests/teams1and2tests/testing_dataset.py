# Ernest Son
# This will construct a testing dataframe with a few random advertising IDs
# This is to meet the <200MB limit on streamlit for development purposes
import pandas as pd
from pygeohash import encode

# Global vars for filtering and getting data
file_list = ["../data/week_data/gps_2018-04-" + (str(n) if n > 9 else "0" + str(n)) + ".csv" for n in range(8, 16)]
adids = ['60670673-2052-f998-ba54-b7ad8a138844',
         '81696261-3059-7d66-69cc-67688182f974',
         '54aa7153-1546-ce0d-5dc9-aa9e8e371f00']
relevant_features = ['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id', 'horizontal_accuracy', 'carrier', 'model', 'wifi_ssid', 'wifi_bssid']
out_data = pd.DataFrame(columns=relevant_features)
for adid in adids:
    for fname in file_list:
        data = pd.read_csv(fname, sep=",")
        parsed_df = data.loc[data.advertiser_id == adid]
        out_data = pd.concat([parsed_df, out_data])
        print('data added...')
out_data['geohash'] = out_data.apply(lambda d: encode(d.latitude, d.longitude, precision=10), axis=1)
out_data = out_data[relevant_features]
out_data.to_csv("../data/test.csv")

