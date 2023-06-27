from typing import Union
import logging
import os

import typer
import mlflow
import pandas as pd

from utils.logging_handler import logging_decorator
from utils.features import Features

app = typer.Typer()

def load_list(
    list_path: str
) -> Union[pd.DataFrame, list]:  

    """This function is used to load the list of urls from a file.

    Args:
        list_path (str): The path to the file that contains the list of urls.
    
    Returns:
        Union[pd.DataFrame, list]: The list of urls.
    """
 
    if list_path.split(".")[-1] == "csv":
        df = pd.read_csv(list_path)
        return df
    elif list_path.split(".")[-1] == "txt":
        with open(list_path, "r") as file:
            urls = file.read().splitlines()
            return urls

def save_list(
    data: Union[pd.DataFrame, list], 
    output_path: str
) -> None:

    """This function is used to save the list of urls to a file.

    Args:
        data (Union[pd.DataFrame, list]): The list of urls.
        output_path (str): The path to the file where the list of urls is going to be saved.
    
    Returns:
        None
    """

    if isinstance(data, pd.DataFrame):
        _, extension = os.path.splitext(output_path)
        if extension == '.csv':
            data.to_csv(output_path, index=False)
        elif extension == '.txt':
            data.to_csv(output_path, index=False, header=False)
  
def load_model(
    experiment_id: str
) -> tuple:

    """This function is used to load the model from MLflow.

    Args:
        experiment_id (str): The id of the experiment that contains the model.

    Returns:    
        tuple: A tuple containing the model and the scaler object.
    """

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
    dataset_path: str,
    experiment_id: str,
    artifact_name: str
) -> None:

    """This function is used to predict the class of a url or a list of urls.

    Args:
        data (Union[str, str]): The url or the path to the file that contains the list of urls.
        output_path (Union[str, list]): The path to the file where the results are going to be saved.
        dataset_path (str): The path to the dataset.
        experiment_id (str): The id of the experiment that contains the model.
        artifact_name (str): The name of the artifact that contains the model.

    Returns:
        None
    """ 

    model, scaler = load_model(experiment_id=experiment_id)

    feature_extraction = Features()

    if data[0] == None and data[1] == None:
        logging.critical("Please provide either a url or a list of urls.")
        return

    if data[0] != None:
        features = feature_extraction.extract_features(url=data[0], source_data=dataset_path)
        features_df = pd.DataFrame([features])
        normalized_features = scaler.transform(features_df)

        output = model.predict(normalized_features)
        logging.info(f"The prediction for the url {data[0]} is {output[0]}.")

    elif data[1] != None:
        urls = load_list(list_path=data[1])

        if isinstance(urls, pd.DataFrame):
            feature_results = urls.apply(lambda row: feature_extraction.extract_features(url=row.iloc[0], source_data=dataset_path), axis=1)
            df = pd.concat([urls, feature_results.apply(pd.Series)], axis=1)
            df.iloc[:, 1:] = scaler.transform(df.iloc[:, 1:])
            df['prediction'] = df.apply(lambda row: model.predict(row[1:].values.reshape(1, -1))[0], axis=1)


            save_list(data=df.iloc[:, [0, -1]], output_path=output_path)
            logging.info(f"Predictions for the list of urls have been saved to {output_path}.")
        
        if isinstance(urls, list):
            df = pd.DataFrame(urls, columns=['url'])

            feature_results = df.apply(lambda row: feature_extraction.extract_features(url=row.iloc[0], source_data=dataset_path), axis=1)
            df = pd.concat([df, feature_results.apply(pd.Series)], axis=1)
            df.iloc[:, 1:] = scaler.transform(df.iloc[:, 1:])
            df['prediction'] = df.apply(lambda row: model.predict(row[1:].values.reshape(1, -1))[0], axis=1)


            save_list(data=df.iloc[:, [0, -1]], output_path=output_path)
            logging.info(f"Predictions for the list of urls have been saved to {output_path}.")

@app.command()
def main(
    url: str = typer.Option(None, "--url", "-u", help="The url of the model that is going to be used for prediction."),
    list_path: str = typer.Option(None, "--list", "-l", help="The list of images that are going to be used for prediction."),
    output_path: str = typer.Option(None, "--output", "-o", help="The output file where the predictions are going to be saved."),
    dataset_path: str = typer.Option("src/data/dataset.csv", "--dataset", "-d", help="The dataset that is going to be used for prediction."),
    experiment_id = typer.Option('165635318050438364', "--experiment_id", "-e", help="The id of the experiment that is going to be used for prediction."),
    artifact_name = typer.Option('xgb_model', "--artifact_name", "-a", help="The name of the artifact that is going to be used for prediction.")
) -> None:

    """This function is used to predict the class of a url or a list of urls.

    Args:
        url (str): The url of the model that is going to be used for prediction.
        list_path (str): The list of images that are going to be used for prediction.
        output_path (str): The output file where the predictions are going to be saved.
        dataset_path (str): The dataset that is going to be used for prediction.
        experiment_id (str): The id of the experiment that is going to be used for prediction.
        artifact_name (str): The name of the artifact that is going to be used for prediction.
    
    Returns:
        None
    """
    
    predict(
        data=(url, list_path), 
        output_path=output_path, 
        dataset_path=dataset_path,
        experiment_id=experiment_id, 
        artifact_name=artifact_name
    )

if __name__ == "__main__":
    app()