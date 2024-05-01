import pandas as pd
from .locations import locations_of_interest
from .people import colocation
from .utils.files import random_line, find, random_name
from .utils.geocode import reverse_geocode
from .prediction import double_cluster, get_cluster_centroids, get_top_N_clusters, fit_predictor, haversine
import math
from agile.utils.tag import find_all_nearby_nodes

class Profile:
    def __init__(self, data: pd.DataFrame, ad_id: int, ext_duration: int = 7,
            rep_duration: int = 24, coloc_duration: int = 12, alias: str = None,
            sd: int = 0, alias_dict: dict = None) -> None:
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
        print('Beginning of LOI:\n',self.lois.head(),'\nEnd of LOI')
        try:
            self.lois = reverse_geocode(self.lois)
            print('GeoCoding Successful')
        except:
            pass
        
        self.alias_dict = alias_dict
        
        self.data, self.model, self.cluster_centroids, self.model_accuracy = data, None, None, None
        
        self.coloc = colocation(data, self.lois, coloc_duration)
        self.coloc_addendum()
        
    def add_location_data(self, data: pd.DataFrame):
        location_data =  find_all_nearby_nodes(data, 10)
        location_data.rename(columns={'lat': 'latitude', 'lon': 'longitude'}, inplace=True)
        data['address'] = data.apply(lambda row: self.find_location(row, location_data), axis=1) 
        print("in ald")
        print(data)
        return data

    def find_location(self, row, df):
        location = df[(df['longitude'] == row['longitude']) & (df['latitude'] == row['latitude'])]['name'].values
        return location[0] if len(location) > 0 else None

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
        """
        Runs the clustering algorithm and the locations of interest algorithm to generate a report. This function tests out a bunch of different parameters for each algorithm
        
        Parameters:
        df (pd.DataFrame): The data where LOIs should be found
        """
        cluster_data = pd.DataFrame()
        loi_data = pd.DataFrame()
        loi_data_i = pd.DataFrame()
        
        # run the cluster algorithm, from 1 to 5 clusters
        for cluster_num in range(4,5):
            print('Running Clustering Algorithm...')
            cluster_data = double_cluster(self.ad_id, df)
            loi_data_i = get_top_N_clusters(cluster_data, cluster_num)
            
            try:
                loi_data_i['i'] = [cluster_num] * len(loi_data_i)
                loi_data_i = self.filter_close_coordinates(loi_data_i, threshold_distance=.7)
                loi_data = pd.concat([loi_data, loi_data_i])
            except:
                return pd.DataFrame()
            
        # run the locations of interest algorithm, covering a 25hrs to 1hr for extended duration and 73hrs to 1hr for repetition duration
        # change to (1,25,3) and (71,0,-10)
        for ext_d in range(1,25,3):
            for rep_d in range(71,0,-10):
                print(f'Running LOI Algorithm (Extended Duration: {ext_d} hrs, Repetition Duration: {rep_d} hrs)')
                loi_data = pd.concat([loi_data,locations_of_interest(df, self.ad_id, ext_d, rep_d)]).reset_index(drop=True)
            
        # Finalize the dataframe with these functions
        loi_data = loi_data.drop_duplicates()    
        loi_data = self.filter_close_coordinates(loi_data, threshold_distance=.4)       
        loi_data = self.add_location_data(loi_data)

        return loi_data
    
    def coloc_addendum(self):
        """
        Improves the colocation IDs dataframe on the sidebar
        """
        
        # Checks if there are any colocated IDs. If there are...
        if self.coloc is not None and len(self.coloc) > 0 and len(self.alias_dict) > 0:
            # Create a dataframe of just unique colocated ADIDs
            colocs_df = pd.DataFrame(self.coloc['advertiser_id'].unique(), columns=['Colocated ADIDs'])
            
            # Create a new column for the alias
            colocs_df['Alias'] = [''] * len(colocs_df)
            
            # Iterate through each row. Add the alias to the correct column
            for adid in colocs_df['Colocated ADIDs'].values:
                '''if None in self.data.query('advertiser_id==@self.ad_id')['advertiser_id_alias'].unique():
                    generated_name = self.alias_dict[adid]
                    self.data.loc[self.data['advertiser_id'] == adid, 'advertiser_id_alias'] = generated_name
                    colocs_df.loc[colocs_df['Colocated ADIDs'] == adid, 'Alias'] = generated_name
                else:
                    colocs_df.loc[colocs_df['Colocated ADIDs'] == adid, 'Alias'] = self.data.query('advertiser_id==@self.ad_id')['advertiser_id_alias'].unique()[0]'''
                colocs_df.loc[colocs_df['Colocated ADIDs'] == adid, 'Alias'] = self.alias_dict[adid]
            
            # Merge with the other colocation dataframe
            self.coloc = pd.merge(left=self.coloc, right=colocs_df, left_on='advertiser_id', right_on='Colocated ADIDs')
            
        else:
            self.coloc = pd.DataFrame(columns=['advertiser_id','Colocated IDs','Alias','latitude','longitude'])
            
        # Get the lat and long of where the colocation was identified and put it in the dataframe  
        self.coloc['lat/long'] = self.coloc['latitude'].astype(str) + ', ' + self.coloc['longitude'].astype(str)
        
        self.coloc = self.coloc.drop_duplicates()
        
        '''for adid in self.coloc['Colocated ADIDs'].values:
            adid_alias = self.alias_dict[adid]
            self.coloc.loc[self.coloc['Colocated ADIDs'] == adid, 'Alias'] = adid_alias'''
        
        print('Beginning of Colocated IDs\n',self.coloc,'\nEnd of Colocated IDs')
                
                    
        
            

