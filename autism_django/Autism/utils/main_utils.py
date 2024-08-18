import yaml
from Autism.exception import AutismException
import sys, os , dill
import numpy as np
import pandas as pd
from Autism.logger import logging
from Autism.constant.training_pipeline import *

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise AutismException(e,sys) from  e


def write_yaml_file(file_path: str, content:object , replace:bool=False) -> None:
    try:
# sourcery skip: merge-nested-ifs
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, "w") as file:
                yaml.dump(content, file)
    except Exception as e:
        raise AutismException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise AutismException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise AutismException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise AutismException(e, sys) from e


def load_object(file_path:str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The File: {file_path} is not exist")
        with open(file_path , "rb") as file_obj:
            obj=dill.load(file_obj)
            return obj

    except Exception as e:
        raise AutismException(e, sys) from e

def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    try:
        dataset_schema = read_yaml_file(schema_file_path)

        data = dataset_schema[DATASET_SCHEMA_COLUMNS_KEYS]


        schema = {list(d.keys())[0]: list(d.values())[0] for d in data}
        print(schema)

        dataframe = pd.read_csv(file_path)

        error_message = ""

        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_message = f"{error_message} \column: [{column}] is not in the schmea."

        if len(error_message) > 0:
            raise Exception(error_message)

        return dataframe

    except Exception as e:
        raise AutismException(e, sys) from e