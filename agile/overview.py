import pandas as pd

def adid_value_counts(df):
    temp_df = pd.DataFrame(df['advertiser_id'].value_counts())
    temp_df.columns = ['Occurences in Data']
    return temp_df
