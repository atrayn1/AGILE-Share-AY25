# Sam Chanow
# Ernest Son
# Class File for Profile Class
# Contains information about an individual advertising ID
# This can contain stuff such as LOIs, timestamped data, etc
from english_words import english_words_set
import random
import pandas as pd
from reportlab.pdfgen.canvas import Canvas
from report_template import PDFPSReporte

class Profile:

    # Constructor
    #Some fake values
    def __init__(self, adId) -> None:
        self.adid = adId
        self.name = self.__name_gen()
        self.lois = self.__loi_gen()
        self.coloc = self.__coloc_gen()
        self.pol = self.__pol_gen()

    # __ means private (thanks python absolute dogshit)
    def __name_gen(self) -> str:
        english = list(english_words_set)
        return random.choice(english) + "-" + random.choice(english)

    # generate the locations of interest for this adID
    # TODO import loi.py
    def __loi_gen(self) -> pd.DataFrame:
        pass

    # generate the colocating adId Dataframe for this adID
    def __coloc_gen(self) -> pd.DataFrame:
        pass

    # generate a pattern of life for this adID
    # probably not gonna be a pandas dataframe but the idea is that we have a
    # member that we can use for comparisons with other instances
    def __pol_gen(self) -> pd.DataFrame:
        pass

    # generate report file for this profile
    def generate_report(self) -> None:
        #report = Canvas(self.name + "_report.pdf")
        #report.drawString(72, 72, self.name)
        #report.save()
        #Pass this object to the report generator
        report = PDFPSReporte('psreport.pdf', self)

sam = Profile("SAM")
sam.generate_report()
