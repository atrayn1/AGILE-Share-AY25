# Class File for Profile Class
# Contains information about an individual advertising ID
# This can contain stuff such as LOIs, timestamped data, etc

# Sam Chanow
# Ernest Son

import random
import pandas as pd
from .locations import locations_of_interest
from .people import colocation
from .utils.files import random_line
from .utils.files import find
from .utils.geocode import reverse_geocode
from .prediction import double_cluster
from .prediction import get_cluster_centroids
from .prediction import get_top_N_clusters
from .prediction import fit_predictor

class Profile:

    # Constructor with specific parameters
    def __init__(self, data, ad_id, ext_duration, rep_duration, coloc_duration) -> None:
        self.ad_id = ad_id
        self.name = self.__name_gen()
        # We need to somehow store this information in here so that it can be relayed on the report
        # These are the default values
        self.ext_duration = ext_duration
        self.rep_duration = rep_duration
        self.coloc_duration = coloc_duration
        # these need to be generated after the parameters are defined
        self.lois = self.__loi_gen(data)
        self.coloc = self.__coloc_gen(data)

        # Prediction model values
        # If the model is untrained it will None
        self.model = None
        self.cluster_centroids = None
        self.model_accuracy = None

    '''
    # Secondary constructor without the colocation and loi information requirements
    # TODO we can figure out how to handle the rest of this later
    def __init__(self, data, ad_id) -> None:
        self.ad_id = ad_id
        self.name = self.__name_gen()
        self.data = data
        # Prediction model values
        # If the model is untrained it will None
        self.model = None
        self.cluster_centroids = None
        self.model_accuracy = None
    '''

    def __name_gen(self) -> str:
        # Updated the open to use the find function, so that file paths are located dynamically
        with open(find('first.txt', '/')) as F, open(find('last.txt', '/')) as L:
            return random_line(F) + '-' + random_line(L)

    # generate the locations of interest for this ad_id
    def __loi_gen(self, data) -> pd.DataFrame:
        lois = locations_of_interest(data, self.ad_id, self.ext_duration, self.rep_duration)
        return reverse_geocode(lois)

    # generate the colocating ad_id Dataframe for this ad_id
    def __coloc_gen(self, data) -> pd.DataFrame:
        return colocation(data, self.lois, self.coloc_duration)
    
    # Public function to see if the profile has a trained model
    def model_trained(self) -> bool:
        return self.model != None
    
    # Train the profile's model on the data provided
    # If no data is provided, it will default to the data provided to the profile 
    # contructor
    def model_train(self, data=None):
        if data == None:
            data = self.data
        clustered_data = double_cluster(self.ad_id, data)
        self.cluster_centroids = get_cluster_centroids(clustered_data)
        self.lois = get_top_N_clusters(clustered_data, 5)
        self.model, self.model_accuracy = fit_predictor(clustered_data, False)
        return self.model, self.model_accuracy

    # Perform a single prediction on the provided time
    # The time must be entered as the number of seconds since midnight on that day
    def model_predict(self, time, day) -> int:
        # Returns the label and the centroid associated
        X = pd.DataFrame([[time, day]], columns=['seconds', 'dayofweek'])
        label = self.model.predict(X)[0]
        # TODO Getting a size mismatch error here
        return label, self.cluster_centroids.loc[self.cluster_centroids['label'] == label]

