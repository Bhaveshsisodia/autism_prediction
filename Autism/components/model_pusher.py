from Autism.logger import logging
from Autism.exception import AutismException
from Autism.entity.artifact_entity import ModelPusherArtifact,\
ModelEvaluationArtifact
from Autism.entity.config_entity import ModelPusherConfig
import os, sys
import shutil


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Model Pusher log started.{'<<' * 30} ")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            raise AutismException(e, sys) from e

    def initiate_model_pusher(self,) -> ModelPusherArtifact:
        try:
            trained_model_path = self.model_evaluation_artifact.evaluated_model_path

            ## Creating Model Directory
            model_file_path=self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=model_file_path)

            #save model dir
            saved_model_path =self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)

            # prepare artifact
            model_pusher_artifact = ModelPusherArtifact(is_model_pusher=True
                                                        ,saved_model_path=saved_model_path,
                                                       model_file_path=model_file_path )

            logging.info(f"Model Pusher Artifact :{model_pusher_artifact}")



            return model_pusher_artifact


        except Exception as e:
            raise AutismException(e,sys)


    def __del__(self):
        logging.info(f"{'>>' * 20}Model Pusher log completed.{'<<' * 20} ")