import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

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
        try :
            
            logging.info("data tranformation intiated")  
            
            
            
               # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z'] 
            
              
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            
            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            
            return preprocessor

            logging.info('Pipeline Completed')      
        
        
        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)
        
        
        
        
        
        
        
        
    def initaite_data_transformation(self,train_path,test_path):
        try :
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            
            
            preprocessing_obj = self.get_data_transformation_object()
            
            
            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
            
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            # x_train , y_train

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            # X_test, y_test
            
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # in this line we have to convert target_feature_train_df in the array from and input_feature_train_arr is already a array
            # because when we do fit_tranfrom or transfrom it will return in form of array 
            # and we are doing nothing just at first we divide the data into train and test and that data we use in this
            # and then we divide data into x:- independant and y:- dependant for both train and test then
            # fit_tranfrom for train and transform for test data of x only means input features then 
            # combine them in a form of array in which train and test array contrain independant tranformed data and y data(input feature
            # and traget feautre)
            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                # file ka path or processor pas kia 

            )
            
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                # transformed train and test data as well as file path for pickle file returned
            )

        
        except Exception as e :
            logging.info("exception occured in this intiate data tranformation")
            raise CustomException(e,sys)    
    
    
        
           