import sys
import os
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Reading of train and test data completed')

            preprocessor_obj = self.get_data_transformation_obj()

            logging.info('Loaded the preprocessor object')

            target = 'math_score'

            input_feature_train = train_df.drop(target,axis=1)
            input_feature_test = test_df.drop(target,axis=1)
            target_feature_train = train_df[target]
            target_feature_test = test_df[target]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test)]

            logging.info('Transformed the training and testing data')

            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessor_obj)

            logging.info('Saved the preprocessing object')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)