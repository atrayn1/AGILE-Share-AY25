#Sam Chanow
#Class File for Profile Class
#Contains information about a specific "individual" advertising ID
#This can contain stuff such as LOIs, timestamped data, etc
from english_words import english_words_set
import random
import pandas as pd

class Profile:

    #Constructor
    #Some fake values
    def __init__(self, AdId) -> None:
        self.adid = AdId
        self.name = self.name_gen()
        self.lois = self.__loi_gen()
        self.coloc = self.__coloc_gen()
    
    #__ means private (thanks python absolute dogshit)
    def __name_gen(self) -> str:
        english = list(english_words_set)
        return random.choice(english) + "-" + random.choice(english)
    
    #generate the locations of interest for this ADID
    def __loi_gen(self) -> pd.Dataframe:
        pass

    #Generate the colocating AdId Dataframe for this AdID
    def __coloc_gen(self) -> pd.DataFrame:
        pass
    
    #Generate report file for this profile
    def generate_report(self, filename) -> None:
        pass
#sam = Profile()
#print(sam.name)