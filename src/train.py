import typer
from typing_extensions import Annotated

app = typer.Typer()

def load_data():
    df = pd.read_csv("src/data/raw/train.csv")
    return df 

def normalize_features():
    pass

def split_train_test():
    pass

def fit_xgboost():
    pass

def hyperopt():
    pass

@app.command()
def main(
    source_dataset: str = typer.Option(False, "--source_data", "-sd", help="The original dataset that is going to be used for training"),
    external_data: str = typer.Option(False, "-external_data", "-ed", help="Extra data that is going to be merged with the source dataset"),
    hyperopt: Annotated[bool, typer.Option("-hyperopt", "-hpe", help="Enables hyperopt within the training")] = False,
    max_evals: int = typer.Option(False, "-max_evals", "-me", help="Max evals that we are going to use for the hyperopt"),
    early_stopping_rounds: int = typer.Option(False, "--early_stopping_rounds", "-esr", help="Early stopping rounds for the xgboost model"),
    gamma: float = typer.Option(False, "-gamma", "-g", help="Gamma value for the xgboost model"),
    max_depth: int = typer.Option(False, "-max_depth", "-md", help="Max depth value for the xgboost model"),
    eta: float = typer.Option(False, "-eta", "-e", help="Eta value for the xgboost model"),
    alpha: float = typer.Option(False, "-alpha", "-a", help="Alpha value for the xgboost model"),
) -> None:

    """This is the main function that is going to be used for training the model

    Args:
        source_dataset (str): The original dataset that is going to be used for training
        external_data (str): Extra data that is going to be merged with the source dataset
        hyperopt (bool): Enables hyperopt within the training
        max_evals (int): Max evals that we are going to use for the hyperopt
        early_stopping_rounds (int): Early stopping rounds for the xgboost model
        gamma (float): Gamma value for the xgboost model
        max_depth (int): Max depth value for the xgboost model
        eta (float): Eta value for the xgboost model
        alpha (float): Alpha value for the xgboost model
    
    Returns:
        None

    """
    pass


if __name__ == "__main__":
    app()