from Autism.constant.training_pipeline import SCHEMA_FILE_PATH
from Autism.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from Autism.entity.config_entity import DataValidationConfig
from Autism.exception import AutismException
from scipy.stats import ks_2samp
from Autism.logger import logging
from Autism.utils.main_utils import read_yaml_file , write_yaml_file
import os, sys
import pandas as pd


class DataValidation:


    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                        data_validation_config:DataValidationConfig):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise AutismException(e, sys) from e

    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise AutismException(e, sys) from e


    def detect_dataset_drift(self,base_df, current_df, threshold=0.05)-> bool:
        try:
            status = True
            report ={}
            # sourcery skip: assign-if-exp, boolean-if-exp-identity, instance-method-first-arg-name, remove-unnecessary-cast, simplify-dictionary-update
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                is_same_dist=ks_2samp(d1,d2)
                if is_same_dist.pvalue > threshold:
                    is_found = True
                    status= False
                else:
                    is_found = False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report, )
            return status
        except Exception as e:
            raise AutismException(e, sys) from e

    def is_numerical_column_exist(self, dataframe:pd.DataFrame) -> bool:
        try:
            numerical_columns=self._schema_config['numerical_columns']
            dataframe_columns = dataframe.columns

            numerical_columns_present = True
            missing_numerical_columns =[]
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_columns_present=False
                    missing_numerical_columns.append(num_column)


            logging.info(f"Missing Numerical columns : [{missing_numerical_columns}]")
            return numerical_columns_present



        except Exception as e:
            raise AutismException(e, sys) from e

    def validate_number_of_column(self, dataframe:pd.DataFrame)-> bool:
        try:
            number_of_columns=len(self._schema_config['columns'])
            logging.info(f"Required Number of Columns : {number_of_columns}")
            logging.info(f"Data Frame has columns: {len(dataframe.columns)}")

            if len(dataframe.columns)== number_of_columns:
                logging.info("validation of Number of Columns is Successfully Done")
                return True
            logging.info("validation of Number of Columns is Failed")
            return False

        except Exception as e:
            raise AutismException(e, sys) from e

    @staticmethod
    def get_pandas_dtype(dtype):
        if dtype == 'int':
            return 'int64'
        elif dtype == 'float':
            return 'float64'
        elif dtype == 'category':
            return 'object'
        else:
            return None


    def validate_data_types(self, dataframe:pd.DataFrame)-> bool:
        try:

            expected_dtypes = {list(col.keys())[0]: self.get_pandas_dtype(list(col.values())[0]) for col in self._schema_config['columns']}
            for column, expected_dtype in expected_dtypes.items():
                if column in dataframe.columns:
                    if dataframe[column].dtype.name != expected_dtype:
                        print(f"Column {column} has type {dataframe[column].dtype.name} but expected {expected_dtype}")
                        return False
                else:
                    print(f"Column {column} not found in DataFrame")
                    logging.info("validation of data types Failed !!!!!")
                    return False
            logging.info("validation of data types of Columns is Successfully Done")
            return True

        except Exception as e:
            raise AutismException(e, sys) from e


    def initiate_data_validation(self)-> DataValidationArtifact:
        try:
            error_message= ""
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)

            # validatte number of columns

            status=self.validate_number_of_column(dataframe=train_dataframe)
            if not status:
                error_message=f"{error_message}Train dataframe does not contain all columns.\n"
            status=self.validate_number_of_column(dataframe=test_dataframe)
            if not status:
                error_message=f"{error_message}Test dataframe does not contain all columns.\n"

            # validate numerical columns

            status = self.is_numerical_column_exist(dataframe=train_dataframe)
            if not status:
                error_message=f"{error_message}Train dataframe does not contain all numerical columns.\n"

            status = self.is_numerical_column_exist(dataframe=test_dataframe)

            if not status:
                error_message=f"{error_message}Test dataframe does not contain all numerical columns.\n"

            status = self.validate_data_types(dataframe=train_dataframe)
            if not status:
                error_message=f"{error_message} Train dataframe data types doesn't match with schema file.\n"

            status = self.validate_data_types(dataframe=test_dataframe)
            if not status:
                error_message=f"{error_message} Test dataframe data types doesn't match with schema file.\n"

            if len(error_message)>0:

                raise Exception(error_message)

            ### let's check data drift

            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)

            data_validation_artifact = DataValidationArtifact(validation_status=status,
                                                              valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                                                              valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                                                              invalid_train_file_path=None,
                                                              invalid_test_file_path=None,
                                                              drift_report_file_path=self.data_validation_config.drift_report_file_path
                                                              )

            logging.info(f"Data Validation artifact :{data_validation_artifact}")
            logging.info("******************************** Data Validation Done Successefully**********************************************\n")

            return data_validation_artifact




        except Exception as e:
            raise AutismException(e, sys) from e



