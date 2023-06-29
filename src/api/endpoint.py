import mlflow.xgboost
import xgboost as xgb
from fastapi import FastAPI
from features import feature_extraction
import pandas as pd

app = FastAPI()

@app.on_event("startup")
def load_model():

    runs = mlflow.search_runs(experiment_ids="phishguard")

    # Sort the runs by accuracy in descending order
    sorted_runs = runs.sort_values(by="metrics.accuracy", ascending=False)

    # Get the best run and its corresponding run ID
    best_run_id = sorted_runs.iloc[0]["run_id"]

    model = mlflow.xgboost.load_model(f"runs:/{best_run_id}/xgb_model")
    top_features = model.get_xgb_params()['top_features']
    scaler = mlflow.sklearn.load_model(f"runs:/{best_run_id}/min_max_scaler")

    model_uri = "runs:/<run_id>/model"
    app.model = mlflow.xgboost.load_model(model_uri)
    app.scaler = scaler
    app.top_features = top_features

@app.post("/predict")
def predict(data: list):

    df = pd.DataFrame(data, columns=['url'])

    feature_results = df.apply(lambda row: feature_extraction.extract_features(url=row.iloc[0], source_data="src/data/dataset.csv", top_features=app.top_features), axis=1)
    df = pd.concat([df, feature_results.apply(pd.Series)], axis=1)
    df.iloc[:, 1:] = app.scaler.transform(df.iloc[:, 1:])
    df['prediction'] = df.apply(lambda row: app.model.predict(row[1:].values.reshape(1, -1))[0], axis=1)
    print(df)
