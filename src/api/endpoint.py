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
    best_run = mlflow.get_run(best_run_id)
    top_features = best_run.data.params['top_features']

    # Load in the models
    model = mlflow.xgboost.load_model(f"{os.getcwd()}/mlruns/654885389741096205/{best_run_id}/artifacts/xgb_model")
    scaler = mlflow.sklearn.load_model(f"{os.getcwd()}/mlruns/654885389741096205/{best_run_id}/artifacts/min_max_scaler")

    # Assign the models to the app
    app.model = model
    app.scaler = scaler
    app.top_features = int(top_features)

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
    filtered_domains = []

    for domain in domains:
        if domain.name.startswith("http://") or domain.name.startswith("https://"):
            if "www." not in domain.name:
                url = domain.name.replace("//", "//www.")
            else:
                url = 'https://www.' + domain.name.split("//www.")[-1] 
                if "www." not in url:
                    url = url.replace("//", "//www.")
        elif domain.name.startswith("www."):
            url = 'https://www.' + domain.name[4:]
        else:
            url = 'https://www.' + domain.name
        filtered_domains.append(url)

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
    
