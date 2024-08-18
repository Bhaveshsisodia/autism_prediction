from Autism.entity.artifact_entity import DataTransformationArtifact , ModelTrainerArtifact
from Autism.entity.config_entity import ModelTrainerConfig
from Autism.exception import AutismException
from Autism.logger import logging
from Autism.utils.main_utils import load_numpy_array_data , save_object , load_object
import os, sys
from Autism.entity.model_factory import MetricInfoArtifact, ModelFactory, GridSearchedBestModel , evaluate_classification_model
from typing import List
import pandas as pd
from xgboost import XGBClassifier
from Autism.ml.model.estimator import AutismModel
# from Autism.ml.metric.classification_metric import get_classification_score





class AutismEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"\n{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config= model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise AutismException(e, sys) from e





    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            logging.info("Loading transformed testing dataset")
            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)


            print(len(pd.DataFrame(train_arr).columns))
            print(len(pd.DataFrame(test_arr).columns))
            logging.info("Splitting training and testing input and target feature")
            x_train, y_train, x_test, y_test = train_arr[:, :- 1], train_arr[:, -1],test_arr[:, :-1], test_arr[:, -1]

            # print(len(pd.DataFrame(x_train).columns))
            # print(len(pd.DataFrame(x_test).columns))
            # print(len(pd.DataFrame(y_train).columns))
            # print(len(pd.DataFrame(y_test).columns))

            logging.info("Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(
                f"Initializing model factory class using above model config file: {model_config_file_path}")
            model_factory = ModelFactory(
                model_config_path=model_config_file_path)

            base_accuracy = self.model_trainer_config.base_accuracy

            logging.info(f"Expected accuracy: {base_accuracy}")

            logging.info("Initiating operation model selecttion")

            best_model = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=base_accuracy)

            logging.info(f"Best model found on training dataset: {best_model}")

            logging.info("Extracting trained model list.")
            grid_searched_best_model_list: List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list

            model_list = [
                model.best_model for model in grid_searched_best_model_list]
            logging.info(
                "Evaluation all trained model on training and testing dataset both"
            )

            metric_info: MetricInfoArtifact = evaluate_classification_model(
                model_list=model_list, X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, base_accuracy=base_accuracy)


            logging.info("Best found model on both training and testing dataset.")

            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_object = metric_info.model_object
            model_name=metric_info.model_name +".pkl"

            trained_model_file_path = self.model_trainer_config.trained_model_file_path

            dir_name = os.path.dirname(trained_model_file_path)

# Join the directory name with the new base name
            trained_model_file_path = os.path.join(dir_name, model_name)
            print(trained_model_file_path)


            autism_model = AutismEstimatorModel(
                preprocessing_object=preprocessing_obj, trained_model_object=model_object)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path, obj=autism_model)


            model_trainer_artifact = ModelTrainerArtifact(is_trained=True, message="Model Trained successfully",
                                                                    trained_model_file_path=trained_model_file_path,
                                                                    train_acc= metric_info.train_accuracy,
                                                                    test_acc = metric_info.train_accuracy,
                                                                    train_prec= metric_info.train_precision,
                                                                    test_prec= metric_info.test_precision,
                                                                    train_recall= metric_info.train_recall,
                                                                    test_recall = metric_info.test_recall,
                                                                    model_accuracy= metric_info.model_accuracy

                                                                    )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            logging.info(f"{'*' * 20} Model Trainer Pipeline completed successfully {'*' * 20}/n")
            return model_trainer_artifact


        except Exception as e:
            raise AutismException(e, sys) from e