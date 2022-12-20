#Sam CHanow

# Simple program that will take in multiple files and an ADID and output a new file filted to just that ADIDs points

import pandas as pd

# Global vars for filtering and getting data
file_list = ["../data/week_data/gps_2018-04-" + (str(n) if n > 9 else "0" + str(n)) + ".csv" for n in range(8, 16)]
adid = "07e28697-2930-a575-6c51-2267182152f8"

out_data = pd.DataFrame(columns=["advertiser_id", "datetime", "latitude", "longitude", "horizontal_accuracy", "carrier", "model", "wifi_ssid", "wifi_bssid"])

for fname in file_list:
    data = pd.read_csv(fname, sep=",")
    parsed_df = data.loc[data.advertiser_id == adid]
    print(parsed_df.head())


    out_data = pd.concat([parsed_df, out_data])
    print(out_data.head())

out_data.to_csv("../data/_" + adid + "_weeklong.csv")
