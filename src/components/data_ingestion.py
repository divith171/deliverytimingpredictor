import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import haversine as hs
import numpy as np
import datetime
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')




## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df=pd.read_csv(os.path.join('notebooks/data','finalTrain.csv'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            

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





            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
        



    
     