import os
from typing import List

import mlflow.xgboost
import xgboost as xgb
from fastapi import FastAPI
import pandas as pd
import mlflow
from pydantic import BaseModel

from src.features import Features


app = FastAPI()

class Domain(BaseModel):
    name: str

class DomainsRequest(BaseModel):
    domains: List[Domain]

class PredictionResponse(BaseModel):
    url: str
    prediction: str

@app.on_event("startup")
def load_model():

    """this function is used to load the model and the scaler from the MLflow run.

    Args:
        None
    
    Returns:
        None
    """
        
    runs = mlflow.search_runs(experiment_ids="654885389741096205")

    # Sort the runs by accuracy in descending order
    sorted_runs = runs.sort_values(by="metrics.accuracy", ascending=False)

    # Get the best run and its corresponding run ID
    best_run_id = sorted_runs.iloc[0]["run_id"]

    # Load in the models
    model = mlflow.xgboost.load_model(f"{os.getcwd()}/mlruns/654885389741096205/{best_run_id}/artifacts/xgb_model")
    top_features = model.get_xgb_params()['top_features']
    scaler = mlflow.sklearn.load_model(f"{os.getcwd()}/mlruns/654885389741096205/{best_run_id}/artifacts/min_max_scaler")

    # Assign the models to the app
    app.model = model
    app.scaler = scaler
    app.top_features = top_features

@app.post("/predict")
def predict(data: DomainsRequest) -> List[PredictionResponse]:

    """This function is used to predict the class of a url or a list of urls.

    Args:
        data (Union[str, str]): The url or the path to the file that contains the list of urls.
    
    Returns:
        List[PredictionResponse]: A list of PredictionResponse objects.
    """

    # Getting the post data
    domains = data.domains

    # Getting the urls from the post data
    filtered_domains = [
        'https://www.' + domain.name if not domain.name.startswith("http://www.") and not domain.name.startswith("https://www.") else domain.name
        for domain in domains
    ]

    # Creating a dataframe with the urls
    df = pd.DataFrame(filtered_domains, columns=['url'])

    # Extracting the features
    feature_extraction = Features()
    feature_results = df.apply(lambda row: feature_extraction.extract_features(url=row.iloc[0], source_data="src/data/dataset.csv", top_features=app.top_features), axis=1)

    # Concatenating the features to the dataframe
    df = pd.concat([df, feature_results.apply(pd.Series)], axis=1)
    df.iloc[:, 1:] = app.scaler.transform(df.iloc[:, 1:])

    # Predicting the class of the urls  
    df['prediction'] = df.apply(lambda row: app.model.predict(row[1:].values.reshape(1, -1))[0], axis=1)

    # Creating the response
    response = [PredictionResponse(url=row['url'], prediction=row['prediction']) for idx, row in df.iterrows()]

    return response
    
