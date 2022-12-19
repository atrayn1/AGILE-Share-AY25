#Sam CHanow

# Simple program that will take in multiple files and an ADID and output a new file filted to just that ADIDs points

import pandas as pd

# Global vars for filtering and getting data
file_list = ["../data/week_data/gps_2018-04-" + (str(n) if n > 9 else "0" + str(n)) + ".csv" for n in range(8, 16)]
adid = "54aa7153-1546-ce0d-5dc9-aa9e8e371f00"

out_data = pd.DataFrame(columns=["advertiser_id", "datetime", "latitude", "longitude", "horizontal_accuracy", "carrier", "model", "wifi_ssid", "wifi_bssid"])

for fname in file_list:
    data = pd.read_csv(fname, sep=",")
    parsed_df = data.loc[data.advertiser_id == adid]
    print(parsed_df.head())


    out_data = pd.concat([parsed_df, out_data])
    print(out_data.head())

out_data.to_csv("../data/_" + adid + "_weeklong.csv")
