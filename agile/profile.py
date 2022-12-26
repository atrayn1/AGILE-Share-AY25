# Class File for Profile Class
# Contains information about an individual advertising ID
# This can contain stuff such as LOIs, timestamped data, etc

# Sam Chanow
# Ernest Son

import random
import pandas as pd
from locations.loi import locations_of_interest as loi
from utils.files import random_line

# testing
from filtering.adid import query_adid
from report import Report

test = pd.DataFrame(columns=['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id'])

class Profile:

    # Constructor
    def __init__(self, adID) -> None:
        self.adid = adID
        self.name = self.__name_gen()
        self.lois = self.__loi_gen()
        self.coloc = self.__coloc_gen()
        self.pol = self.__pol_gen()
        self.overpass = self.__overpass_gen()
        #We need to somehow store this information in here so that it can be relayed on the report
        #These are the default values
        self.prec = 10
        self.ext_duration = 7
        self.rep_duration = 24

    # __ means private (thanks python absolute dogshit)
    def __overpass_gen(self) -> pd.DataFrame:
        return test

    def __name_gen(self) -> str:
        with open('../names/first.txt') as f, open('../names/last.txt') as l:
            return random_line(f) + '-' + random_line(l)

    # generate the locations of interest for this adID
    def __loi_gen(self) -> pd.DataFrame:
        return test

    # generate the colocating adID Dataframe for this adID
    def __coloc_gen(self) -> pd.DataFrame:
        return test

    # generate a pattern of life for this adID
    # probably not gonna be a pandas dataframe but the idea is that we have a
    # member that we can use for comparisons with other instances
    def __pol_gen(self) -> pd.DataFrame:
        return test

# test report
'''
df = pd.read_csv("../data/_54aa7153-1546-ce0d-5dc9-aa9e8e371f00_weeklong.csv")
data = query_adid("54aa7153-1546-ce0d-5dc9-aa9e8e371f00", df)
ubl = Profile("54aa7153-1546-ce0d-5dc9-aa9e8e371f00")
ubl.lois = loi(data, 10, 7, 24)
Report(ubl)
'''

