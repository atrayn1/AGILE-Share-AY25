# Class File for Profile Class
# Contains information about an individual advertising ID
# This can contain stuff such as LOIs, timestamped data, etc

# Sam Chanow
# Ernest Son

import random
import pandas as pd
from locations.loi import locations_of_interest
from people.colocation import colocation
from utils.files import random_line
from utils.geocode import reverse_geocode

class Profile:

    # Constructor with specific parameters
    def __init__(self, data, ad_id, prec, ext_duration, rep_duration, coloc_duration) -> None:
        self.ad_id = ad_id
        self.name = self.__name_gen()
        # We need to somehow store this information in here so that it can be relayed on the report
        # These are the default values
        self.prec = prec
        self.ext_duration = ext_duration
        self.rep_duration = rep_duration
        self.coloc_duration = coloc_duration
        # these need to be generated after the parameters are defined
        self.lois = self.__loi_gen(data)
        self.coloc = self.__coloc_gen(data)

    def __name_gen(self) -> str:
        with open('../names/first.txt') as F, open('../names/last.txt') as L:
            return random_line(F) + '-' + random_line(L)

    # generate the locations of interest for this ad_id
    def __loi_gen(self, data) -> pd.DataFrame:
        lois = locations_of_interest(data, self.ad_id, self.prec, self.ext_duration, self.rep_duration)
        return reverse_geocode(lois)

    # generate the colocating ad_id Dataframe for this ad_id
    def __coloc_gen(self, data) -> pd.DataFrame:
        return colocation(data, self.lois, self.coloc_duration)

# test report
from report import Report
data = pd.read_csv("../data/weeklong_gh.csv")
ubl = Profile(data, "54aa7153-1546-ce0d-5dc9-aa9e8e371f00", 10, 4, 24, 2)
Report(ubl)
print("report generated!")

