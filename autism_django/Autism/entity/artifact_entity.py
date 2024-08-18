from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


# @dataclass
# class ClassificationMetricArtifact:
#     f1_score: float
#     precision_score: float
#     recall_score: float

@dataclass
class ModelTrainerArtifact:
    is_trained: bool
    message: str
    trained_model_file_path : str
    train_acc: float
    test_acc : float
    train_prec: float
    test_prec: float
    train_recall: float
    test_recall : float
    model_accuracy: float



@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    message: str
    evaluated_model_path : str



@dataclass
class ModelPusherArtifact:
    is_model_pusher:str
    saved_model_path:str
    model_file_path:str


