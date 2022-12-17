import pandas as pd

data = pd.read_csv("../data/test_location_data_gh.csv")
one_id = data.loc[data['advertiser_id'] == '54aa7153-1546-ce0d-5dc9-aa9e8e371f00'].reset_index().to_csv('one_id.csv')
