import pandas as pd
from .locations import locations_of_interest
from .people import colocation
from .utils.files import random_line, find
from .utils.geocode import reverse_geocode
from .prediction import double_cluster, get_cluster_centroids, get_top_N_clusters, fit_predictor

class Profile:
    def __init__(self, data: pd.DataFrame, ad_id: int, ext_duration: int = 7, rep_duration: int = 24, coloc_duration: int = 2) -> None:
        """
        Constructs a Profile object with given parameters.

        Parameters:
        data (pd.DataFrame): Dataframe containing data for multiple advertising IDs.
        ad_id (int): Advertising ID for which the profile is being created.
        ext_duration (int): Time period (in days) for which locations of interest will be considered for the profile.
        rep_duration (int): Time period (in hours) for which locations of interest will be considered repeatedly for the profile.
        coloc_duration (int): Time period (in hours) for which other advertising IDs co-located with this advertising ID will be considered for the profile.
        """
        self.ad_id = ad_id
        self.name = self.__name_gen()
        self.ext_duration, self.rep_duration, self.coloc_duration = ext_duration, rep_duration, coloc_duration
        self.lois = reverse_geocode(locations_of_interest(data, ad_id, ext_duration, rep_duration))
        self.coloc = colocation(data, self.lois, coloc_duration)
        self.data, self.model, self.cluster_centroids, self.model_accuracy = data, None, None, None

    def __name_gen(self) -> str:
        """
        Generates a random name for the profile.
        """
        with open(find('../names/first.txt')) as F, open(find('../names/last.txt')) as L:
            return random_line(F) + '-' + random_line(L)

    def model_trained(self) -> bool:
        """
        Checks whether the profile's model is trained or not.

        Returns:
        bool: True if the model is trained, False otherwise.
        """
        return self.model is not None
    
    def model_train(self, data: pd.DataFrame = None) -> tuple:
        """
        Trains the profile's model on the provided data.

        Parameters:
        data (pd.DataFrame, optional): Dataframe containing data for multiple advertising IDs. If not provided, the profile's default data is used.

        Returns:
        tuple: A tuple containing the trained model and its accuracy score.
        """
        if data is None:
            data = self.data
        clustered_data = double_cluster(self.ad_id, data)
        self.cluster_centroids = get_cluster_centroids(clustered_data)
        self.lois = get_top_N_clusters(clustered_data, 5)
        self.model, self.model_accuracy = fit_predictor(clustered_data, False)
        return self.model, self.model_accuracy

    def model_predict(self, time: int, day: int) -> tuple:
        """
        Performs a single prediction on the provided time.

        Parameters:
        time (int): The time of the day (in seconds since midnight).
        day (int): The day of the week (0-6, where 0 is Monday).

        Returns:
        tuple: A tuple containing the predicted label and its associated centroid.
        """
        X = pd.DataFrame([[time, day]], columns=['seconds', 'dayofweek'])
        label = self.model.predict(X)[0]
        centroid = self.cluster_centroids.loc[self.cluster_centroids['label'] == label]
        return label, centroid

