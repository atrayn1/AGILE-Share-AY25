# Simple program that will take in multiple files and outputs a new file that combines all the relevant features

import pandas as pd

# Global vars for filtering and getting data
file_list = ["../data/week_data/gps_2018-04-" + (str(n) if n > 9 else "0" + str(n)) + ".csv" for n in range(8, 16)]

relevant_features = ["advertiser_id", "datetime", "latitude", "longitude"]
out_data = pd.DataFrame(columns=relevant_features)

for fname in file_list:
    data = pd.read_csv(fname, sep=",", usecols=relevant_features)
    out_data = pd.concat([data, out_data])
    print(out_data.head())

out_data.to_csv("../data/weeklong.csv")
