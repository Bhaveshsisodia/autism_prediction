# Autism Prediction End-to-End Project
Autism is a neurological disorder that affects a person’s ability to interact with others, make eye contact with others, learn capacity, and other behavioral and social capabilities of a person.
But there is no ascertain way to tell whether a person has Autism or not because there are no such diagnostics methods available to diagnose this disorder. so with help of machine learning we will predict the disorder or not.

web page
## Overview
This project is an end-to-end pipeline for predicting autism spectrum disorder (ASD) using machine learning techniques. The pipeline encompasses multiple stages, including data ingestion, data validation, data transformation, model training, model evaluation, and model deployment. Additionally, the project includes hyperparameter tuning to optimize model performance. The final model is deployed to a Django-based web application, enabling users to perform predictions and retrain the model with new data.

## Pipeline Components
1. <strong>Data Ingestion</strong> :<br>
The Data Ingestion module is responsible for gathering and storing data from various sources. This module can handle different data formats and stores the ingested data in a structured format suitable for further processing.

2. <strong>Data Validation</strong> :<br>
The Data Validation module ensures that the ingested data meets the required quality standards. It checks for missing values, invalid data types, outliers, and other anomalies. The module generates validation reports that provide insights into the data quality.

3. <strong>Data Transformation</strong> :<br>
The Data Transformation module processes the validated data and prepares it for model training. This includes scaling, encoding categorical variables, handling missing values, feature engineering, and other data preprocessing steps.

4. <strong> Model Trainer</strong> :<br>
The Model Trainer module is responsible for training the machine learning model. It uses the transformed data to train the model, and automatically performs hyperparameter tuning to optimize the model's performance.

5. <strong> Model Evaluation</strong> :<br>
The Model Evaluation module evaluates the trained model using various metrics such as accuracy, precision, recall, F1-score, etc. It compares the performance of different models and selects the best one for deployment.

6. <strong>Model Pusher</strong> :<br>
The Model Pusher module deploys the trained model to production. It saves the model in a specified location and prepares it for integration with the web application.

7. <strong>Web Application (Django)</strong> :<br>
The Web Application is built using Django and provides a user-friendly interface for making predictions and retraining the model. Users can upload new data, trigger the retraining process, and view prediction results through the web interface.

8. <strong>Hyperparameter tunning</strong> :<br>
The project includes automatic hyperparameter tuning to find the best combination of parameters for the machine learning model. This is done using techniques like Grid Search.


## How to Run the Project

1. Clone the Repository
    ```
    git clone https://github.com/Bhaveshsisodia/autism_prediction.git
    cd autism_prediction
    ```

2. Install Dependencies
    ```
    pip install -r requirements.txt
    ```

3. Start the Django Web Application
    ```
    cd web_app
    python manage.py runserver
    ```
4. Access the web application at http://127.0.0.1:8000.

    ```
    for Training: http://127.0.0.1:8000/train
    for Prediction : http://127.0.0.1:8000/predict


