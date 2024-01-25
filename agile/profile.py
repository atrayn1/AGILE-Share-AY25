import pandas as pd
from .locations import locations_of_interest
from .people import colocation
from .utils.files import random_line, find
from .utils.geocode import reverse_geocode
from .prediction import double_cluster, get_cluster_centroids, get_top_N_clusters, fit_predictor, haversine
import math

class Profile:
    def __init__(self, data: pd.DataFrame, ad_id: int, ext_duration: int = 7,
            rep_duration: int = 24, coloc_duration: int = 12, alias: str = None,
            sd: int = 0) -> None:
        """
        Constructs a Profile object with given parameters.

        Parameters:
        data (pd.DataFrame): Dataframe containing data for multiple advertising IDs.
        ad_id (int): Advertising ID for which the profile is being created.
        ext_duration (int): Time period (in days) for which locations of interest will be considered for the profile.
        rep_duration (int): Time period (in hours) for which locations of interest will be considered repeatedly for the profile.
        coloc_duration (int): Time period (in hours) for which other advertising IDs co-located with this advertising ID will be considered for the profile.
        alias (str): A given alias made by the user already for an ADID
        """
        self.ad_id = ad_id
        self.sd = sd
        self.name = self.__name_gen() if alias == None else alias
        self.ext_duration, self.rep_duration, self.coloc_duration = ext_duration, rep_duration, coloc_duration
        '''try:
            # Check if Nominatim is down, and if it is do not reverse Geocode the data (allows for most algorithms to run more reliably)
            self.lois = reverse_geocode(locations_of_interest(data, ad_id, ext_duration, rep_duration))
        except:
            self.lois = locations_of_interest(data, ad_id, ext_duration, rep_duration)'''
           
        
        self.lois = self.create_report_lois(data)
        print(self.lois.head())
        try:
            self.lois = reverse_geocode(self.lois)
            print('GeoCoding Successful')
        except:
            pass
        
        print(self.lois.head())
        
        self.coloc = colocation(data, self.lois, coloc_duration)
        self.data, self.model, self.cluster_centroids, self.model_accuracy = data, None, None, None
    
    def reliability(self) -> float:
        n = len(self.cluster_centroids)
        acc = self.model_accuracy

        return acc * math.log(n) / math.log(301)

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

        print(len(self.cluster_centroids))
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
    
    def filter_close_coordinates(self, df: pd.DataFrame = None, threshold_distance: float = 0.13):
        """
        Filters out any LOI locations that are within threshold distance of one another
        
        Parameters:
        df (pd.DataFrame): the data to be filtered
        threshold_distance (float): how close two data points can be without being considered too close, in kilometers
        
        Returns:
        pd.DataFrame: the filtered DataFrame
        """
        new_df = pd.DataFrame(columns=df.columns)

        for i, row1 in df.iterrows():
            keep_row = True
            for _, row2 in new_df.iterrows():
                distance = haversine(row1['latitude'], row1['longitude'], row2['latitude'], row2['longitude'])
                if distance < threshold_distance:
                    keep_row = False
                    break
            if keep_row:
                new_df = new_df.append(row1, ignore_index=True)

        return new_df
    
    def create_report_lois(self, df: pd.DataFrame):
        cluster_data = pd.DataFrame()
        loi_data = pd.DataFrame()
        loi_data_i = pd.DataFrame()
        
        # run the cluster algorithm, from 1 to 5 clusters
        for cluster_num in range(1,5):
            print(cluster_num)
            cluster_data = double_cluster(self.ad_id, df)
            loi_data_i = get_top_N_clusters(cluster_data, cluster_num)
            loi_data_i['i'] = [cluster_num] * len(loi_data_i)
            loi_data_i = self.filter_close_coordinates(loi_data_i, threshold_distance=.7)
            loi_data = pd.concat([loi_data, loi_data_i])
            
        # run the locations of interest algorithm, covering a 25hrs to 1hr for extended duration and 73hrs to 1hr for repetition duration
        for ext_d in range(1,25,3):
            for rep_d in range(73,0,-4):
                print(ext_d,rep_d)
                loi_data = pd.concat([loi_data,locations_of_interest(df, self.ad_id, ext_d, rep_d)]).reset_index(drop=True)
            
        loi_data = loi_data.drop_duplicates()    
            
        loi_data = self.filter_close_coordinates(loi_data, threshold_distance=.4)       

        return loi_data
        
            

