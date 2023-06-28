<p align="center">
  <a style="font-weight: bold;">Phishing site detection based on several classification models</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python Version">
  <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/UncoveredTensor/phishguard">
  <img alt="GitHub Discussions" src="https://img.shields.io/github/discussions/UncoveredTensor/phishguard">
  <img alt="GitHub issues" src="https://img.shields.io/github/issues/UncoveredTensor/phishguard">
</p>

---

PhishGuard is a robust tool designed to classify whether a webpage is a phishing site or not. This evaluation relies on advanced classification models that are trained on diverse open-source datasets, referenced in our resource header. Our aim with this project is to gather data about potential phishing sites. By doing so, we aim to create public awareness and help individuals determine the safety of a webpage, thereby protecting them from harmful online experiences. (Note: Although we have a well-trained dataset to train the models effectively, extracting features from the URL remains complicated due to the lack of instructions regarding the context and specific details for each feature. As a result, we improvised the feature extraction process, which can potentially result in false positives.)

# Install Phishguard

<details close>
<summary>Docker</summary>
<br>
  
    docker pull SOON!
</details>

## Usage
```
Usage:
  pipenv run predict --help
  
Flags:
PREDICTION:
  -u, --url     A url that is going to be used for prediction.
  -l, --list    A file path where a batch of urls are within it.

OUTPUT:
  -o, --output   The output path where the predictions are going to be saved.

DATASET:
  -d, --dataset 
  
MODEL:
  -e, --experiment_id                      The id of the experiment that is going to be used for prediction.
  -ma, --model_artifact_name               The name of the model artifact that is going to be used for prediction.
  -sa, --min_max_scaler_artifact_name      The name of the min_max_scaler artifact that is going to be used for normalization.
```

## Dependencies

To ensure consistent dependencies within the project, we recommend running the following pipenv command to lock all the dependencies used in the project. This revision maintains clarity and provides a concise instruction, showing the importance of using the pipenv command to lock dependencies for consistency.

    pipenv install

## Retrain
Since the project will continously be updated with new data and be retrained by UncoveredTensor, we give the option also to contributors and developers to retrain the model to their liking. Note: it is nice to know that the data that could be added on should be the same kind of features we want to use, since the model that is being used takes a certain feature matrix as input.

To retrain the model architecture we can use the help command.

```
Usage:
  pipenv run train --help
  
Flags:
DATASET:
  -sd, --source_data       A path to the original dataset that is going to be used for training.
  -ed, --external_data     A path to an extra data source that is going to be merged with the source dataset.
  -tf, --top_features      The amount of top features we wanna use based on correlation matrix.
  
MODEL:
  -ma, --model_artifact_name             The name of the min max scaler artifact that is going to be saved.
  -sa, --min_max_scaler_artifact_name    The name of the min max scaler artifact that is going to be saved.                                                                                              
  -ts, --train-size                      The train dataset size for when splitting the data into train and validation set.
  -esr, --early_stopping_rounds          The amount of patience within the early stopping function.
  -g, --gamma                            The gamma range for training the model. (0.1, 1) - default (0.1, 1.5).
  -md, --max_depth                       The max depth range for training the model (1, 50) - default (1, 101).
  -e, --eta                              The learning range thats going to be used when training the model (-4, 0), default (-3, 0).
  -a, --alpha                            The alpha range thats going to be used within the training of the model (0.01, 0.8), default (0.01, 1.0).
  
HYPEROPT:
  -hpe, --hyperopt         Defines whether hyperopt needs to be used when training the XGBoost model.
  -me, --max_evals         The amount of models we wanna train when doing hyperopt on our model default (50)
```

## Example 

Below this text, you will find example usages for classifying whether a page is a phishing site or not. You can perform classifications using two approaches: either by analyzing a large batch of pages or by evaluating a single website individually.

To classify a single website, you can use the following command:

    pipenv run phishguard -u "EXAMPLE_URL"

To classify a batch of websites, you can use the following command:

    pipenv run phishguard -l [example.csv or example.txt] -o [output path, csv or txt]



## Inference
The model can be executed in inference mode by utilizing the Docker image specified in this README. Upon launching the Docker container, the API can be accessed conveniently from localhost:3000.

# For Contributors
For those of you interested in contributing to the PhishGuard project, we highly appreciate your involvement. Whether you're a data scientist, a web developer, or someone with an interest in cybersecurity, there are plenty of opportunities for you to help improve PhishGuard. Underneath the this text we can see the sub headers where we can find the resources we have used within this project.

## Resources
in the "Resources" section of the project, you'll find direct links to the main components that have been instrumental in building the project. This includes links to the open-source datasets we used for training our model and the XGBoost model that forms the backbone of our project's functionality. Additionally, you'll find a link to the tracking server that helps monitoring and managing the model's performance. These resources offer a comprehensive overview for anyone interested in understanding or contributing to the project.

XGBoost: **https://xgboost.readthedocs.io/en/stable/)https://xgboost.readthedocs.io/en/stable/** <br>
Dataset: **https://data.mendeley.com/datasets/72ptz43s9v/1** <br>
Tracking Server: (COMING SOON!)
