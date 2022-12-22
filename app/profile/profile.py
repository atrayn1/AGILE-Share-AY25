#Sam Chanow
#Class File for Profile Class
#Contains information about a specific "individual" advertising ID
#This can contain stuff such as LOIs, timestamped data, etc
from english_words import english_words_set
import random
import pandas as pd
from reportlab.pdfgen.canvas import Canvas

class Profile:

    #Constructor
    #Some fake values
    def __init__(self, AdId) -> None:
        self.adid = AdId
        self.name = self.__name_gen()
        self.lois = self.__loi_gen()
        self.coloc = self.__coloc_gen()
    
    #__ means private (thanks python absolute dogshit)
    def __name_gen(self) -> str:
        english = list(english_words_set)
        return random.choice(english) + "-" + random.choice(english)
    
    #generate the locations of interest for this ADID
    def __loi_gen(self) -> pd.DataFrame:
        pass

    #Generate the colocating AdId Dataframe for this AdID
    def __coloc_gen(self) -> pd.DataFrame:
        pass
    
    #Generate report file for this profile
    def generate_report(self) -> None:
        report = Canvas(self.name + "_report.pdf")
        report.drawString(72, 72, self.name)
        report.save()

sam = Profile("SAM")
sam.generate_report()