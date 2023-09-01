import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import haversine as hs
import numpy as np
import datetime


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 ID:object,
                 Delivery_person_ID:str,
                 Delivery_person_Age:float,
                 Delivery_person_Ratings:float,
                 Restaurant_latitude:float,
                 Restaurant_longitude:float,
                 Delivery_location_latitude:float,Delivery_location_longitude:float,Order_Date:str,Time_Orderd:str,
                 Time_Order_picked:str,Weather_conditions:str,Road_traffic_density:str,Vehicle_condition:int,
                 Type_of_order:str,Type_of_vehicle:str,multiple_deliveries:float,Festival:str,City:str,
                 ):
        
        self.ID=ID
        self.Delivery_person_ID=Delivery_person_ID
        self.Delivery_person_Age=Delivery_person_Age
        self.Delivery_person_Ratings=Delivery_person_Ratings
        self.Restaurant_latitude=Restaurant_latitude
        self.Restaurant_longitude=Restaurant_longitude
        self.Delivery_location_latitude = Delivery_location_latitude
        self.Delivery_location_longitude = Delivery_location_longitude
        self.Order_Date = Order_Date
        self.Time_Orderd=Time_Orderd
        self.Time_Order_picked=Time_Order_picked
        self.Weather_conditions=Weather_conditions
        self.Road_traffic_density=Road_traffic_density
        self.Vehicle_condition=Vehicle_condition
        self.Type_of_order=Type_of_order
        self.Type_of_vehicle=Type_of_vehicle
        self.multiple_deliveries=multiple_deliveries
        self.Festival=Festival
        self.City=City
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'ID':[self.ID],
                'Delivery_person_ID':[self.Delivery_person_ID],
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Restaurant_latitude':[self.Restaurant_latitude],
                'Restaurant_longitude':[self.Restaurant_longitude],
                'Delivery_location_latitude':[self.Delivery_location_latitude],
                'Delivery_location_longitude':[self.Delivery_location_longitude],
                'Order_Date':[self.Order_Date],
                'Time_Orderd':[self.Time_Orderd],
                'Time_Order_picked':[self.Time_Order_picked],
                'Weather_conditions':[self.Weather_conditions],
                'Road_traffic_density':[self.Road_traffic_density],
                'Vehicle_condition':[self.Vehicle_condition],
                'Type_of_order':[self.Type_of_order],
                'Type_of_vehicle':[self.Type_of_vehicle],
                'multiple_deliveries':[self.multiple_deliveries],
                'Festival':[self.Festival],
                'City':[self.City]

            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            
            df=df.drop(labels=['Delivery_person_Age'],axis=1)
            df=df.drop(labels=['Delivery_person_ID'],axis=1)
            df=df.drop(labels=['Order_Date'],axis=1)
            df=df.drop(labels=['ID'],axis=1)
            # Custom transformer to calculate haversine distance
            class HaversineDistance(BaseEstimator, TransformerMixin):
                def fit(self, Z, y=None):
                    return self
                
                def transform(self, Z):
                    hav=[]
                    for i in range(len(df)):
                        loc1=(Z.iloc[i,1],Z.iloc[i,2])
                        loc2=(Z.iloc[i,3],Z.iloc[i,4])
                        hav.append(hs.haversine(loc1,loc2))
                    
                    Z['Distance']=hav
                    Z['Distance']=Z['Distance'].round(2)
                    Z=Z.drop(labels=['Restaurant_latitude'],axis=1)
                    Z=Z.drop(labels=['Restaurant_longitude'],axis=1)
                    Z=Z.drop(labels=['Delivery_location_longitude'],axis=1)
                    Z=Z.drop(labels=['Delivery_location_latitude'],axis=1)
                    
                    return Z
            
            
            # Custom transformer to split time columns and handle missing values
            class TimeSplitter(BaseEstimator, TransformerMixin):
                def fit(self, Z, y=None):
                    return self
                
                def transform(self, Z):
                    
                    converted_datetimes = []
                    for value in Z['Time_Orderd']:
                        if isinstance(value, str):  # Check if the value is a string
                            try:
                                converted_time = datetime.datetime.strptime(value, '%H:%M').time()
                                converted_datetime = datetime.datetime.combine(datetime.datetime.min, converted_time)
                            except ValueError:
                                converted_datetime = None
                        else:
                            converted_datetime = None  # Handle NaN values
                        converted_datetimes.append(converted_datetime)

                    Z['time_column'] = converted_datetimes
                    # Extract numerical features from datetime.datetime column
                    Z['Orderd_hour'] = Z['time_column'].apply(lambda x: x.hour if x else None)
                    Z['Orderd_minute'] = Z['time_column'].apply(lambda x: x.minute if x else None)

                    # Drop the original datetime.datetime column
                    Z.drop(columns=['time_column'], inplace=True)


                    converted_datetimes1 = []
                    for value in Z['Time_Order_picked']:
                        if isinstance(value, str):  # Check if the value is a string
                            try:
                                converted_time = datetime.datetime.strptime(value, '%H:%M').time()
                                converted_datetime = datetime.datetime.combine(datetime.datetime.min, converted_time)
                            except ValueError:
                                converted_datetime = None
                        else:
                            converted_datetime = None  # Handle NaN values
                        converted_datetimes1.append(converted_datetime)

                    Z['Altered_Time_Order_picked'] = converted_datetimes1
                    # Extract numerical features from datetime.datetime column
                    Z['Orderd_picked_hour'] = Z['Altered_Time_Order_picked'].apply(lambda x: x.hour if x else None)
                    Z['Orderd_picked_minute'] = Z['Altered_Time_Order_picked'].apply(lambda x: x.minute if x else None)
                    Z.drop(columns=['Altered_Time_Order_picked'], inplace=True)


                    for index, row in Z.iterrows():
                        if pd.isna(row['Orderd_hour']) and pd.isna(row['Orderd_minute']):
                            picked_time_in_minutes = row['Orderd_picked_hour'] * 60 + row['Orderd_picked_minute']
                            adjusted_time_in_minutes = picked_time_in_minutes - 15
                            
                            adjusted_hour = adjusted_time_in_minutes // 60
                            adjusted_minute = adjusted_time_in_minutes % 60
                            
                            Z.at[index, 'Orderd_hour'] = adjusted_hour
                            Z.at[index, 'Orderd_minute'] = adjusted_minute

                    for index, row in Z.iterrows():
                        if pd.isna(row['Orderd_picked_hour']) and pd.isna(row['Orderd_picked_minute']):
                            picked_time_in_minutes = row['Orderd_hour'] * 60 + row['Orderd_minute']
                            adjusted_time_in_minutes = picked_time_in_minutes - 15
                            
                            adjusted_hour = adjusted_time_in_minutes // 60
                            adjusted_minute = adjusted_time_in_minutes % 60
                            
                            Z.at[index, 'Orderd_picked_hour'] = adjusted_hour
                            Z.at[index, 'Orderd_picked_minute'] = adjusted_minute


                    column_to_check = 'Orderd_hour'
                    Z = Z.dropna(subset=[column_to_check])
                    Z=Z.drop(labels=['Time_Orderd'],axis=1)
                    Z=Z.drop(labels=['Time_Order_picked'],axis=1)
                    
                    return Z


            # Create a combined preprocessing pipeline
            combined_pipeline = Pipeline([
                ('haversine', HaversineDistance()),
                ('time_splitter', TimeSplitter()),
            
            ])
            df = combined_pipeline.transform(df)
            
            
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)