import mlflow.xgboost
import xgboost as xgb
from fastapi import FastAPI
from src.features import Features
import pandas as pd
from mlflow.tracking import MlflowClient
import mlflow

app = FastAPI()

@app.on_event("startup")
def load_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000") # replace with your mlflow service URL
    client = MlflowClient()

    # Replace 'YourExperimentID' with your actual experiment id and 'accuracy' with the metric you are interested in.
    experiment_id = "phishguard"
    runs = client.search_runs(experiment_ids=experiment_id,
                              filter_string="",
                              run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                              max_results=1000,
                              order_by=["metrics.accuracy DESC"])

    # Get the top run (with the highest accuracy)
    top_run = runs[0]

    model = mlflow.xgboost.load_model(f"runs:/{top_run.info.run_id}/xgb_model")
    top_features = model.get_xgb_params()['top_features']
    scaler = mlflow.sklearn.load_model(f"runs:/{top_run.info.run_id}/min_max_scaler")

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
