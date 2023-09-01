import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
import numpy as np
import datetime
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            
            numerical_cols=['Delivery_person_Ratings', 'Vehicle_condition', 'multiple_deliveries',
                    'Distance', 'Orderd_hour', 'Orderd_minute', 'Orderd_picked_hour','Orderd_picked_minute'] 
            ordinal_cols = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order','Type_of_vehicle', 'Festival', 'City']

            # Define the custom ranking for each ordinal variable
            Weather_conditions_map={"Sunny":1,"Cloudy":2,"Windy":3,"Fog":4,"Stormy":5,"Sandstorms":6}
            Road_traffic_density_map={"Low":1,"Medium":2,"High":3,"Jam":4}
            Type_of_order_map={"Drinks":1,"Snack":2,"Meal":3,"Buffet":4}
            Type_of_vehicle_map={"motorcycle":1,"scooter":2,"electric_scooter":3,"bicycle":4}
            City_map={"Semi-Urban":1,"Urban":2,"Metropolitian":3}
            Festival_map={"No":0,"Yes":1}                            

            logging.info('Pipeline Initiated')
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            ordinal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories=[list(Weather_conditions_map.keys()), list(Road_traffic_density_map.keys()), list(Type_of_order_map.keys()),list(Type_of_vehicle_map.keys()),
                list(Festival_map.keys()),list(City_map.keys())])),
                ('scaler', StandardScaler())
            ])
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('ordinal_encoder', ordinal_pipeline, ordinal_cols)
            ])


            return preprocessor

            logging.info('Pipeline Completed')




        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        

    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name]
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)