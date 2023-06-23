<p align="center">
  <a style="font-weight: bold;">Phishing site detection based on several classification models</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue.svg" alt="Python Version">
  <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/UncoveredTensor/phishguard">
  <img alt="GitHub Discussions" src="https://img.shields.io/github/discussions/UncoveredTensor/phishguard">
  <img alt="GitHub issues" src="https://img.shields.io/github/issues/UncoveredTensor/phishguard">
</p>

---

PhishGuard is a robust tool designed to classify whether a webpage is a phishing site or not. This evaluation relies on advanced classification models that are trained on diverse open-source datasets, referenced in our resource header. Our aim with this project is to gather data about potential phishing sites. By doing so, we aim to create public awareness and help individuals determine the safety of a webpage, thereby protecting them from harmful online experiences.

# Install Phishguard

<details close>
<summary>Docker</summary>
<br>
  
    docker pull SOON!
</details>

## Usage

## Dependencies

To ensure consistent dependencies within the project, we recommend running the following pipenv command to lock all the dependencies used in the project. This revision maintains clarity and provides a concise instruction, showing the importance of using the pipenv command to lock dependencies for consistency.

    pipenv install

## Example 

Below this text, you will find example usages for classifying whether a page is a phishing site or not. You can perform classifications using two approaches: either by analyzing a large batch of pages or by evaluating a single website individually.

To classify a single website, you can use the following command:

    pipenv run phishguard -u "EXAMPLE_URL"

To classify a batch of websites, you can use the following command:

    pipenv run phishguard -b [example.csv or example.txt]

## Inference
The model can be executed in inference mode by utilizing the Docker image specified in this README. Upon launching the Docker container, the API can be accessed conveniently from localhost:3000.

# For Contributors
For those of you interested in contributing to the PhishGuard project, we highly appreciate your involvement. Whether you're a data scientist, a web developer, or someone with an interest in cybersecurity, there are plenty of opportunities for you to help improve PhishGuard. Underneath the this text we can see the sub headers where we can find the resources we have used within this project.

## Resources
