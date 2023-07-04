from typing import Union, Callable, Tuple
import logging
import os

import typer
import mlflow
import pandas as pd

from utilities.logging_handler import logging_decorator
from features import Features

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

        df = pd.read_csv(list_path, header=None)

        potential_header = df.iloc[0, 0]

        logging.info("Reading the csv file and loading in the urls")

        if "." not in potential_header:
            df.columns = [potential_header]
            df = df.iloc[1:, :]
            return df
        else:
            df.columns = ['url']
            return df

    elif list_path.split(".")[-1] == "txt":
        with open(list_path, "r") as file:
            urls = file.read().splitlines()
            logging.info("Reading the txt file and loading in the urls")
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
            logging.info(f"Saved the list of urls and predictions to {output_path}")
        elif extension == '.txt':
            data.to_csv(output_path, index=False, header=False)
            logging.info(f"Saved the list of urls and predictions to {output_path}")
  
def load_model(
    experiment_id: str,
    model_artifact_name: str,
    min_max_scaler_artifact_name: str
) -> tuple:

    """This function is used to load the model from MLflow.

    Args:
        experiment_id (str): The id of the experiment that contains the model.

    Returns:    
        tuple: A tuple containing the model and the scaler object.
    """

    runs = mlflow.search_runs(experiment_ids=experiment_id)
    logging.info("Getting the runs of the tracking server")

    # Sort the runs by accuracy in descending order
    sorted_runs = runs.sort_values(by="metrics.accuracy", ascending=False)
    logging.info("Sorting the runs by accuracy in descending order")

    # Get the best run and its corresponding run ID
    best_run_id = sorted_runs.iloc[0]["run_id"]
    logging.info("Getting the best run ID")

    model = mlflow.xgboost.load_model(f"runs:/{best_run_id}/{model_artifact_name}")
    top_features = model.get_xgb_params()['top_features']
    scaler = mlflow.sklearn.load_model(f"runs:/{best_run_id}/{min_max_scaler_artifact_name}")
    logging.info("Loading the model and the scaler object")

    return model, scaler, top_features

def url_predict(
    feature_extraction: Callable,
    data: str,
    source_data: str,
    top_features: int,
    scaler: Callable,
    model: Callable
) -> Tuple[str, int]:
    
    """This function is used to predict the class of a url.

    Args:
        feature_extraction (Callable): The function that is used to extract the features from the url.
        data (str): The url.
        source_data (str): The path to the dataset.
        top_features (int): The number of features that are going to be used for the prediction.
        scaler (Callable): The scaler object that is used to normalize the features.
        model (Callable): The model that is used to make the prediction.
        
    Returns:
        tuple: A tuple containing the url and the prediction.
    
    """

    features = feature_extraction.extract_features(url=data[0], source_data=source_data, top_features=top_features)

    logging.info("Getting the features of the url")
    features_df = pd.DataFrame([features])

    normalized_features = scaler.transform(features_df)
    logging.info("Normalizing the features")

    output = model.predict(normalized_features)
    logging.info("Making the prediction")

    return data, output

def list_predict(
    feature_extraction: Callable,
    urls: Union[pd.DataFrame, list],
    source_data: str,
    output_path: str,
    top_features: int,
    scaler: Callable,
    model: Callable
):
    
    """this function is used to predict the class of a list of urls.

    Args:
        feature_extraction (Callable): The function that is used to extract the features from the urls.
        urls (Union[pd.DataFrame, list]): The list of urls.
        source_data (str): The path to the dataset.
        output_path (str): The path to the file where the results are going to be saved.
        top_features (int): The number of features that are going to be used for the prediction.
        scaler (Callable): The scaler object that is used to normalize the features.

    Returns:
        str: The path to the file where the results are saved.    
    """
    
    if isinstance(urls, list):
        urls = pd.DataFrame(urls, columns=['url'])
    
    feature_results = urls.apply(lambda row: feature_extraction.extract_features(url=row.iloc[0], source_data=source_data, top_features=top_features), axis=1)
    df = pd.concat([urls, feature_results.apply(pd.Series)], axis=1)
    df.iloc[:, 1:] = scaler.transform(df.iloc[:, 1:])
    df['prediction'] = df.apply(lambda row: model.predict(row[1:].values.reshape(1, -1))[0], axis=1)
    save_list(data=df.iloc[:, [0, -1]], output_path=output_path)
        
    return output_path

@logging_decorator(project_name="phishguard")
def predict(
    data: Union[str, str], 
    output_path: Union[str, list], 
    dataset_path: str,
    experiment_id: str,
    model_artifact_name: str,
    min_max_scaler_artifact_name: str
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

    model, scaler, top_features = load_model(
        experiment_id=experiment_id, 
        model_artifact_name=model_artifact_name,
        min_max_scaler_artifact_name=min_max_scaler_artifact_name  
    )

    feature_extraction = Features()

    if data[0] == None and data[1] == None:
        logging.critical("Please provide either a url or a list of urls.")
        return

    if data[0] != None:

        data, output = url_predict(
            feature_extraction=feature_extraction, 
            data=data, 
            source_data=dataset_path, 
            top_features=top_features, 
            scaler=scaler,
            model=model
        )

        logging.info(f"The prediction for the url {data[0]} is {output[0]}.")

    elif data[1] != None:
        urls = load_list(list_path=data[1])

        output = list_predict(
            feature_extraction=feature_extraction,
            urls=urls,
            source_data=dataset_path, 
            output_path=output_path,
            top_features=top_features, 
            scaler=scaler, 
            model=model
        )

        logging.info(f"Predictions for the list of urls have been made.")

@app.command()
def main(
    url: str = typer.Option(None, "--url", "-u", help="A url that is going to be used for prediction."),
    list_path: str = typer.Option(None, "--list", "-l", help="A file path where a batch of urls are within it."),
    output_path: str = typer.Option(None, "--output", "-o", help="The output path where the predictions are going to be saved."),
    dataset_path: str = typer.Option("src/data/dataset.csv", "--dataset", "-d", help="The dataset we need inorder to get the top features out."),
    experiment_id = typer.Option('654885389741096205', "--experiment_id", "-e", help="The id of the experiment that is going to be used for prediction."),
    model_artifact_name = typer.Option('xgb_model', "--model_artifact_name", "-ma", help="The name of the model artifact that is going to be used for prediction."),
    min_max_scaler_artifact_name = typer.Option('min_max_scaler', "--min_max_scaler_artifact_name", "-sa", help="The name of the min_max_scaler artifact that is going to be used for normalization.")
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
        model_artifact_name=model_artifact_name,
        min_max_scaler_artifact_name=min_max_scaler_artifact_name
    )

if __name__ == "__main__":
    app()