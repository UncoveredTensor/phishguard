import mlflow.xgboost
import xgboost as xgb
from fastapi import FastAPI
from src.features import Features
import pandas as pd
import mlflow
import os

app = FastAPI()

@app.on_event("startup")
def load_model():
    runs = mlflow.search_runs(experiment_ids="654885389741096205")

    # Sort the runs by accuracy in descending order
    sorted_runs = runs.sort_values(by="metrics.accuracy", ascending=False)

    # Get the best run and its corresponding run ID
    best_run_id = sorted_runs.iloc[0]["run_id"]

    print(os.getcwd())

    model = mlflow.xgboost.load_model(f"mlruns/0/654885389741096205/{best_run_id}/artifacts/xgb_model")
    top_features = model.get_xgb_params()['top_features']
    scaler = mlflow.sklearn.load_model(f"mlruns/0/654885389741096205/{best_run_id}/artifacts/min_max_scaler")

    app.model = model
    app.scaler = scaler
    app.top_features = top_features

@app.post("/predict")
def predict(data: list):

    df = pd.DataFrame(data, columns=['url'])

    feature_results = df.apply(lambda row: Features.extract_features(url=row.iloc[0], source_data="src/data/dataset.csv", top_features=app.top_features), axis=1)
    df = pd.concat([df, feature_results.apply(pd.Series)], axis=1)
    df.iloc[:, 1:] = app.scaler.transform(df.iloc[:, 1:])
    df['prediction'] = df.apply(lambda row: app.model.predict(row[1:].values.reshape(1, -1))[0], axis=1)
    print(df)
