from django.shortcuts import render
from django.http import JsonResponse
from Autism.pipeline.training_pipeline import TrainPipeline
from Autism.ml.model.estimator import ModelResolver

from Autism.constant.training_pipeline import SAVED_MODEL_DIR
from Autism.utils.main_utils import load_object
from django.http import HttpResponse
import pandas as pd
from Autism.utils.main_utils import read_yaml_file
from Autism.constant.training_pipeline import SCHEMA_FILE_PATH


def train_route(request):
    try:
        if request.method == 'POST':
            train_pipeline = TrainPipeline()
            if train_pipeline.is_pipeline_running:
                return JsonResponse("Training pipeline is already running.", safe=False)

            train_pipeline.run_pipeline()
            return JsonResponse("Training successful!", safe=False)
        else:
            return JsonResponse("Invalid request method", safe=False)

    except Exception as e:
        return JsonResponse(f"Error Occurred! {e}", safe=False)

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

def predict(request):
    try:
        if request.method == 'POST':
            if 'file' not in request.FILES:
                return JsonResponse("No file uploaded", safe=False)

            # Read the uploaded file into a DataFrame
            file = request.FILES['file']
            df = pd.read_csv(file)
            _schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            df=df.drop(_schema_config["drop_columns"],axis=1)
            df['ageGroup'] = df['age'].apply(convertAge)

            if df.empty:
                return JsonResponse("Uploaded file is empty", safe=False)
            model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
            if not model_resolver.is_model_exist():
                return JsonResponse("Model is not available", safe=False)
            best_model_path = model_resolver.get_best_model_path()
            model = load_object(file_path=best_model_path)
            y_pred = model.predict(df)
            df['predicted_column'] = y_pred

            df_html = df.to_html(classes="table table-striped", index=False)

            # Render the table in the template
            return render(request, 'index.html', {'table': df_html})
        else:
            return JsonResponse("Invalid request method", safe=False)

    except Exception as e:
        raise JsonResponse(f"Error Occured! {e}", safe=False)


def index(request):
    return render(request, 'index.html')