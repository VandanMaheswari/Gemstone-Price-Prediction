import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
# data classes is used for creating the class varaible without any funcationality
# https://docs.python.org/3/library/dataclasses.html


from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('artifacts','train.csv') 
    test_data_path:str = os.path.join('artifacts','test.csv') 
    raw_data_path:str = os.path.join('artifacts','raw.csv') 
    # that is how we can define variable for class without init by using dataclass
    # in this code the os.path means current directory and then join means the folder artifacts and inside this select train.csv
    
    

## create a class for Data Ingestion

class DataIngestion:
    def __init__(self):
        self.ingesion_config = DataIngestionconfig()
            # now inside this variable there is three pass stored
            
    
    def initiate_data_ingesion(self):
        logging.info("Data Ingesion method starts")
        
        try :
            # df = pd.read_csv("notebooks/data","gemstone.csv")
            df = pd.read_csv(r"F:\study material\Data Science\lecture material\50.9th-april\Gemstone-price-prediction\notebooks\data\gemstone.csv")
            logging.info("dataset read as pandas dataframe")
            
            
            
            os.makedirs(os.path.dirname(self.ingesion_config.raw_data_path),exist_ok=True)
            # it is used for creating a directory in which raw data should be store and the name of the file is raw.csv as we give in 
            # DataIngesionconfig class and this will be created inside the artifacts folder
            
             # later i have to test somethin by commenting df train_set and test_set line so we can se if makdir create a dir alomg
            # with create raw.csv file or not 
            
            
            df.to_csv(self.ingesion_config.raw_data_path,index=False)
            # stroing the raw data inside the created artifact folder as raw.csv name 
           
            
            
            logging.info("test test split")
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)
            
            train_set.to_csv(self.ingesion_config.train_data_path,index = False)
            test_set.to_csv(self.ingesion_config.test_data_path,index = False)
            # i am not again pass command of madirs because it is already created so i just have to pass the Train_data_path or
            # test_data_path and it will automatically go and save my files inside artifacts folder as it is already created above
            # while saving raw data
            
            logging.info('Ingestion of Data is completed')
            
            
            return (
                self.ingesion_config.train_data_path,
                self.ingesion_config.test_data_path
            )
        
        
        
       
       
       
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
        
        
        
        
        
