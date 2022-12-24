# Class File for Profile Class
# Contains information about an individual advertising ID
# This can contain stuff such as LOIs, timestamped data, etc

# Sam Chanow
# Ernest Son

from english_words import english_words_set
import random
import pandas as pd
from reportlab.pdfgen.canvas import Canvas

#from report_template import PDFPSReport
from report import Report

class Profile:

    # Constructor
    #Some fake values
    def __init__(self, adId) -> None:
        self.adid = adId
        self.name = self.__name_gen()
        self.lois = self.__loi_gen()
        self.coloc = self.__coloc_gen()
        self.pol = self.__pol_gen()
        self.overpass = self.__overpass_gen()

    def __overpass_gen(self) -> pd.DataFrame:
        return pd.DataFrame(columns=['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']) # Just for testing
    # __ means private (thanks python absolute dogshit)
    def __name_gen(self) -> str:
        english = list(english_words_set)
        return random.choice(english) + "-" + random.choice(english)

    # generate the locations of interest for this adID
    # TODO import loi.py
    def __loi_gen(self) -> pd.DataFrame:
        return pd.DataFrame(columns=['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']) # Just for testing

    # generate the colocating adId Dataframe for this adID
    def __coloc_gen(self) -> pd.DataFrame:
        return pd.DataFrame(columns=['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']) # Just for testing

    # generate a pattern of life for this adID
    # probably not gonna be a pandas dataframe but the idea is that we have a
    # member that we can use for comparisons with other instances
    def __pol_gen(self) -> pd.DataFrame:
        return pd.DataFrame(columns=['geohash', 'datetime', 'latitude', 'longitude', 'advertiser_id']) # Just for testing

    # generate report file for this profile
    def generate_report(self) -> None:
        df = pd.DataFrame({'geohash':['asdf','asdf','asdf'], 'datetime':['mon','tue','wed'], 'latitude':[69, 70, 71], 'longitude':[420, 421, 422], 'advertiser_id':['ubl', 'ubl', 'ubl']})
        #report = PDFPSReport('report.pdf', self)
        report = Report('report.pdf', df)

ubl = Profile("ubl")
ubl.generate_report()

