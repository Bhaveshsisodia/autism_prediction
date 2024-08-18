import numpy as np
import pandas as pd
import sys
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from Autism.constant.training_pipeline import TARGET_COLUMN , SCHEMA_FILE_PATH
from Autism.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from Autism.entity.config_entity import DataTransformationConfig
from Autism.exception import AutismException
from Autism.logger import logging
# from Autism.ml.model.estimator import TargetValueMapping
from Autism.utils.main_utils import save_numpy_array_data, save_object
from Autism.utils.main_utils import read_yaml_file , write_yaml_file
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler


# Custom transformer to apply LabelEncoder on multiple columns
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {col: LabelEncoder() for col in columns}

    def fit(self, X, y=None):
        for col in self.columns:
            self.encoders[col].fit(X[col])
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = self.encoders[col].transform(X[col])
        return X


# top 10 one hot encoding technique due to huge number of data label are present

class TopCategoriesOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=8):
        self.top_n = top_n
        self.top_categories_ = {}
        self.onehot_encoders_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            top_categories = X[col].value_counts().index[:self.top_n]

            self.top_categories_[col] = top_categories

            self.onehot_encoders_[col] = OneHotEncoder(categories=[top_categories], handle_unknown='ignore')

            self.onehot_encoders_[col].fit(X[[col]])
        return self

    def transform(self, X):
        transformed_columns = []
        for col in X.columns:
            transformed_col = self.onehot_encoders_[col].transform(X[[col]]).toarray()
            col_names = [f"{col}_{cat}" for cat in self.top_categories_[col]]
            transformed_df = pd.DataFrame(transformed_col, columns=col_names, index=X.index)
            transformed_columns.append(transformed_df)
        return pd.concat(transformed_columns, axis=1)


class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact,
                    data_transformation_config: DataTransformationConfig,):
        """

        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            logging.info("******************************************** Data Transformation Initiated ***************************")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise AutismException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise AutismException(e, sys)

    @staticmethod
    def convertAge(age):
        if age < 4:
            return 'Toddler'
        elif age < 12:
            return 'Kid'
        elif age < 18:
            return 'Teenager'
        elif age < 40:
            return 'Young'
        else:
            return 'Senior'





    def get_data_transformer_object(self)-> ImbPipeline:

        try:
            label_encoding_columns = self._schema_config['label_encoding']
            one_hot_encoding_columns =self._schema_config['one_hot_encoding']
            median_impute_columns = self._schema_config['median_impute']
            mode_impute_columns = self._schema_config['mode_impute']
            scaling_columns = median_impute_columns + mode_impute_columns
            print(label_encoding_columns,one_hot_encoding_columns,median_impute_columns,scaling_columns)

            label_pipeline = Pipeline([
                    ('label_encoder', MultiColumnLabelEncoder(columns=label_encoding_columns)),
                    ('scaler_1', MinMaxScaler())

                ])

            one_hot_pipeline = Pipeline([
                ('one_hot_encoder', TopCategoriesOneHotEncoder(top_n=10)),
                ('scaler_2', MinMaxScaler())
            ])

            median_pipeline = Pipeline([
                ('median_imputer', SimpleImputer(strategy='median')),
                ('scaler_3', MinMaxScaler())
            ])

            mode_pipeline = Pipeline([
                ('mode_imputer', SimpleImputer(strategy='constant')),
                ('scaler_4', MinMaxScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('label', label_pipeline, label_encoding_columns),
                    ('one_hot', one_hot_pipeline, one_hot_encoding_columns),
                    ('median_impute', median_pipeline, median_impute_columns),
                    ('mode_impute', mode_pipeline, mode_impute_columns)

                ],
                remainder='passthrough'  # Keep other columns as is
            )

            # preprocessor = Pipeline(steps=[
            #             ('preprocessor', pre_processor),
            #             ('oversampler', RandomOverSampler(random_state=42))
            #         ])

            logging.info(f"Data Transformation preprocessor successfully created :{preprocessor}")


            return preprocessor


        except Exception as e:
            raise AutismException(e, sys)

    def initiate_data_transformer(self,) -> DataTransformationArtifact:
        try:
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)

            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            train_df['ageGroup'] = train_df['age'].apply(self.convertAge)
            test_df['ageGroup'] = test_df['age'].apply(self.convertAge)

            preproccesor = self.get_data_transformer_object()

            # training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

             #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            print(len(input_feature_train_df.columns),len(input_feature_test_df.columns))


            preprocessor_object =preproccesor.fit(input_feature_train_df)
            transformed_input_train_feature =preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)


            input_feature_train_final , target_feature_train_final  = RandomOverSampler().fit_resample(transformed_input_train_feature, target_feature_train_df)
            input_feature_test_final , target_feature_test_final = RandomOverSampler().fit_resample(transformed_input_test_feature, target_feature_test_df)

            print(len((pd.DataFrame(input_feature_train_final)).columns), len((pd.DataFrame(input_feature_test_final)).columns))

            # pd.DataFrame(input_feature_train_final).to_csv(r"D:\web Development using python\autism_prediction\artifact\08_08_2024_17_21_35\data_transformation\transformed\input_feature_train_final.csv",index=False)
            # pd.DataFrame(input_feature_test_final).to_csv(r"D:\web Development using python\autism_prediction\artifact\08_08_2024_17_21_35\data_transformation\transformed\input_feature_test_final.csv",index=False)


            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            print(len((pd.DataFrame(train_arr)).columns), len((pd.DataFrame(test_arr)).columns))

            # save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr,)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr,)

            save_object(self.data_transformation_config.transformed_object_file_path, preproccesor,)
            data_transformation_artifact = DataTransformationArtifact(transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data Transformation artifact created Successfully:{data_transformation_artifact}")
            logging.info("************************************************Data Transformation Done Successfully *************************************************\n")
            return data_transformation_artifact






        except Exception as e:
            raise AutismException(e, sys)




