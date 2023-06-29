# Use a base image
FROM python:3.10-slim-buster

# Install Git
RUN apt-get update && \
    apt-get install -y git

# Set the working directory
WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install mlflow 
RUN pip3 install fastapi
RUN pip3 install uvicorn
RUN pip3 install utils

# Clone the GitHub repository
RUN git clone https://github.com/UncoveredTensor/phishguard && \
    cd phishguard && \
    pip install pipenv && \
    pipenv install --system --deploy --ignore-pipfile

WORKDIR /app/phishguard