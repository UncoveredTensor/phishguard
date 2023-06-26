from typing import Tuple, Dict

import warnings
import typer
from typing_extensions import Annotated
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK
from xgboost import XGBClassifier
import mlflow

warnings.filterwarnings("ignore", category=UserWarning)

app = typer.Typer()

def load_data(source_data: str) -> pd.DataFrame:

    """This function is used to load the data from the source.

    Returns:
        pd.DataFrame: The data that is going to be used for training.
    """

    return pd.read_csv(source_data)

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:

    """This function is used to normalize the features.

    Args:
        features (pd.DataFrame): The features that are going to be normalized.
    
    Returns:
        pd.DataFrame: The normalized features.
    """

    df.iloc[:, :-1] = df.iloc[:, :-1] / 255

    return df

def merge_data(df: pd.DataFrame, external_data: str) -> pd.DataFrame:
    
    external_data_df = pd.read_csv(external_data)

    return pd.concat([df, external_data_df], axis=0)

def split_train_test(df: pd.DataFrame, train_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """This function is used to split the data into train and test.

    Args:
        df (pd.DataFrame): The data that is going to be splitted.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The splitted data.
    """

    X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], train_size = 0.8)

    train_set = pd.DataFrame(X_train).assign(y=Y_train)
    test_set = pd.DataFrame(X_test).assign(y=Y_test)

    return train_set, test_set

def get_model_loss_plot(model) -> plt.figure:

    """This function is used to get the model loss plot.

    Args:
        model (object): The model that is going to be evaluated.
    
    Returns:
        plt.figure: The model loss plot.
    """

    results = model.evals_result()
    train_logloss = results['validation_0']['logloss']
    test_logloss = results['validation_1']['logloss']
    epochs = range(1, len(train_logloss) + 1)

    data = {'Epochs': epochs,
            'Train Log Loss': train_logloss,
            'Test Log Loss': test_logloss}

    df = pd.DataFrame(data)

    fig, ax = plt.subplots() 

    sns.lineplot(data=df, x='Epochs', y='Train Log Loss', label='Train', ax=ax)
    sns.lineplot(data=df, x='Epochs', y='Test Log Loss', label='Test', ax=ax)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log Loss')
    ax.set_title('XGBoost Log Loss per Epoch')
    ax.legend()
    plt.close()

    return fig

def get_model_evaluation(model, x_test, y_test) -> Dict[str, float]:

    """This function is used to get the evaluation metrics of the model.

    Args:
        model (object): The model that is going to be evaluated.
        x_test (pd.DataFrame): The test data.
        y_test (pd.DataFrame): The test labels.
    
    Returns:
        Dict[str]: The evaluation metrics.
    """

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return {
        "accuracy": accuracy,
        "log_loss": logloss,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }

def fit_xgboost(hyperspace: dict) -> Dict[float, STATUS_OK]: 

    """This function is used to fit the xgboost model.

    Args:   
        hyperspace (dict): The hyperparameters that are going to be used for the xgboost model.
    
    Returns:
        dict: The loss and the status of the model.
    """

    params = {k: v for k, v in hyperspace.items() if k not in ["train_set", "test_set"]}
    
    X_train = hyperspace['train_set'].iloc[:, :-1]
    y_train = hyperspace['train_set'].iloc[:, -1]
    
    X_test = hyperspace['test_set'].iloc[:, :-1]
    y_test = hyperspace['test_set'].iloc[:, -1]
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    model = XGBClassifier(**params)

    mlflow.set_experiment("phishing")

    with mlflow.start_run():
    
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        evaluation_metrics = get_model_evaluation(model=model, x_test=X_test, y_test=y_test)

        mlflow.xgboost.log_model(model, "xgb_model")

        mlflow.log_params(params)
        
        for metric_name, metric_value in evaluation_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        figure = get_model_loss_plot(model=model)

        mlflow.log_figure(figure, "log_loss.png")

    mlflow.end_run()
        
    return {'loss': evaluation_metrics['log_loss'], 'status': STATUS_OK}

@app.command()
def main(
    source_dataset: str = typer.Option('src/data/dataset.csv', "--source_data", "-sd", help="The original dataset that is going to be used for training."),
    external_data: str = typer.Option(None, "-external_data", "-ed", help="Extra data that is going to be merged with the source dataset."),
    top_features: int = typer.Option(40, "-top_features", "-tf", help="Top features that are going to be used for training."),
    hyperopt: Annotated[bool, typer.Option("-hyperopt", "-hpe", help="Enables hyperopt within the training.")] = False,
    max_evals: int = typer.Option(50, "-max_evals", "-me", help="Max evals that we are going to use for the hyperopt."),
    train_size: float = typer.Option(0.8, "-train_size", "-ts", help="Train size for the train test split."),
    early_stopping_rounds: int = typer.Option(10, "--early_stopping_rounds", "-esr", help="Early stopping rounds for the xgboost model."),
    gamma: Annotated[Tuple[int, int], typer.Option('-gamma', '-g', help="Gamma value for the xgboost model.")] = (0.1, 1.5),
    max_depth: Annotated[Tuple[int, int], typer.Option('-max_depth', '-md', help="Max depth value for the xgboost model.")] = (1, 101),
    eta: Annotated[Tuple[int, int], typer.Option('-eta', '-e', help="Eta value for the xgboost model.")] = (-3, 0),
    alpha: Annotated[Tuple[int, int], typer.Option('-alpha', '-a', help="Alpha value for the xgboost model.")] = (0.01, 1.0),
) -> None:

    """This is the main function that is going to be used for training the model.

    Args:
        source_dataset (str): The original dataset that is going to be used for training.
        external_data (str): Extra data that is going to be merged with the source dataset.
        top_features (int): Top features that are going to be used for training.
        hyperopt (bool): Enables hyperopt within the training.
        max_evals (int): Max evals that we are going to use for the hyperopt.
        train_size (float): Train size for the train test split.
        early_stopping_rounds (int): Early stopping rounds for the xgboost model.
        gamma (float): Gamma value for the xgboost model.
        max_depth (int): Max depth value for the xgboost model.
        eta (float): Eta value for the xgboost model.
        alpha (float): Alpha value for the xgboost model.
    
    Returns:
        None.
    """

    df = load_data(source_data=source_dataset)

    if external_data is not None:
        df = merge_data(df, external_data)
    
    df = normalize_features(df=df)

    columns = df.corr()['phishing'].sort_values(ascending=False)[:top_features].index

    df = df[columns]

    train_set, test_set = split_train_test(df=df, train_size=train_size)
    
    hyperspace = {
        'gamma': hp.uniform('gamma', gamma[0], gamma[1]),
        'eta': hp.loguniform('eta', eta[0], eta[1]),
        'max_depth': hp.randint('max_depth', max_depth[0], max_depth[1]),
        'alpha': hp.uniform('alpha', alpha[0], alpha[1]),
        'train_set': train_set,
        'test_set': test_set,
        'top_features': top_features,
        'objective': 'binary:logistic',
        'early_stopping_rounds': early_stopping_rounds,
        'predictor': 'gpu_predictor'
    }

    if hyperopt:
        best = fmin(fit_xgboost, hyperspace, algo=tpe.suggest, max_evals=max_evals)
    else:
        best = fmin(fit_xgboost, hyperspace, algo=tpe.suggest, max_evals=1)


if __name__ == "__main__":
    app()