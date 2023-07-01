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

@app.on_event("startup")
def load_model():
    runs = mlflow.search_runs(experiment_ids="654885389741096205")

    # Sort the runs by accuracy in descending order
    sorted_runs = runs.sort_values(by="metrics.accuracy", ascending=False)

    # Get the best run and its corresponding run ID
    best_run_id = sorted_runs.iloc[0]["run_id"]

    model = mlflow.xgboost.load_model(f"{os.getcwd()}/mlruns/654885389741096205/{best_run_id}/artifacts/xgb_model")
    top_features = model.get_xgb_params()['top_features']
    scaler = mlflow.sklearn.load_model(f"{os.getcwd()}/mlruns/654885389741096205/{best_run_id}/artifacts/min_max_scaler")

    app.model = model
    app.scaler = scaler
    app.top_features = top_features

@app.post("/predict")
def predict(data: DomainsRequest):

    domains = data.domains

    print(domains)

    df = pd.DataFrame(domains, columns=['url'])

    feature_extraction = Features()
    feature_results = df.apply(lambda row: feature_extraction.extract_features(url=row.iloc[0], source_data="src/data/dataset.csv", top_features=app.top_features), axis=1)
    df = pd.concat([df, feature_results.apply(pd.Series)], axis=1)
    df.iloc[:, 1:] = app.scaler.transform(df.iloc[:, 1:])
    df['prediction'] = df.apply(lambda row: app.model.predict(row[1:].values.reshape(1, -1))[0], axis=1)
    print(df)
