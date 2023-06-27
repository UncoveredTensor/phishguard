from typing import Union
import logging

import typer
import mlflow
import pandas as pd

from utils.logging_handler import logging_decorator
from utils.features import Features

app = typer.Typer()

def load_list(
    list_path: str
):
    with open(list, "r") as f:
        return f.read().splitlines()

def save_list(
    output_type: str,
    output_path: str,
):
    pass

def load_model(
    experiment_id: str
):

    runs = mlflow.search_runs(experiment_ids=experiment_id)

    # Sort the runs by accuracy in descending order
    sorted_runs = runs.sort_values(by="metrics.accuracy", ascending=False)

    # Get the best run and its corresponding run ID
    best_run_id = sorted_runs.iloc[0]["run_id"]

    model = mlflow.xgboost.load_model(f"runs:/{best_run_id}/xgb_model")
    scaler = mlflow.sklearn.load_model(f"runs:/{best_run_id}/min_max_scaler")

    return model, scaler

@logging_decorator("Prediction")
def predict(
    data: Union[str, str], 
    output_path: Union[str, list], 
    output_type: str,
    dataset_path: str,
    experiment_id: str,
    artifact_name: str
):
    
    model, scaler = load_model(experiment_id=experiment_id)

    if data[0] == None and data[1] == None:
        logging.critical("Please provide either a url or a list of urls.")
        return

    if data[0] != None:
        feature_extraction = Features()
        features = feature_extraction.extract_features(url=data[0], source_data=dataset_path)
        features = {key.split("get_", 1)[1]: value for key, value in features.items()}
        features_df = pd.DataFrame([features])
        normalized_features = scaler.transform(features_df)

        output = model.predict(normalized_features)
        logging.info(f"The prediction for the url {data[0]} is {output[0]}.")

    elif data[1] != None:
        urls = load_list(list_path=data[1])
        features_extraction = Features()
        features = features_extraction.extract_features(url=urls[0], source_data=dataset_path)
        features = {key.split("get_", 1)[1]: value for key, value in features.items()}
        features_df = pd.DataFrame([features])
        normalized_features = scaler.transform(features_df)

        output = model.predict(normalized_features)
        logging.info(f"The prediction for the url {data[0]} is {output[0]}.")


@app.command()
def main(
    url: str = typer.Option(None, "--url", "-u", help="The url of the model that is going to be used for prediction."),
    list_path: str = typer.Option(None, "--list", "-l", help="The list of images that are going to be used for prediction."),
    output_path: str = typer.Option(None, "--output", "-o", help="The output file where the predictions are going to be saved."),
    output_type: str = typer.Option(None, "--output_type", "-t", help="The type of the output file where the predictions are going to be saved."),
    dataset_path: str = typer.Option("src/data/dataset.csv", "--dataset", "-d", help="The dataset that is going to be used for prediction."),
    experiment_id = typer.Option('165635318050438364', "--experiment_id", "-e", help="The id of the experiment that is going to be used for prediction."),
    artifact_name = typer.Option('xgb_model', "--artifact_name", "-a", help="The name of the artifact that is going to be used for prediction.")
):
    
    predict(
        data=(url, list_path), 
        output_path=output_path, 
        output_type=output_type,
        dataset_path=dataset_path,
        experiment_id=experiment_id, 
        artifact_name=artifact_name
    )

if __name__ == "__main__":
    app()