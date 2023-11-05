# Machine Learning Zoomcamp Midterm Project

## Overview

The aim of the project is to build a model to predict the bankruptcy of a company given financial indicators. We build a XGBoost Classifier on a selected subset of features and deploy a web service locally with Docker. The main difficulties of the problem:

- We are in the case of an imbalance data classification problem, i.e. the dataset with the skewed target class. Only 3% of observations correspond to bankrupt companies.
- There are too many (96) features. Many of them correlate with each other.

We overcome the problem of imbalanced data using the combination of over-sampling and under-sampling methods. In particular, we use [SMOTEENN](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html). To deal with the number of features, we apply the feature selection methods (mutual information and F value) and remove highly correlated features.

More details on data and on the solution can be found in `notebook.ipynb` jupyter notebook.

The dataset is collected from the UC Irvine Machine Learning Repository. Interested readers can find the short description of the dataset and description of variables in [Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction).

### Environment setup

To manage the environment setup we use `pipenv` virtualenv management tool. We need to execute the following commands:

- `pip install pipenv`
- `pipenv shell`
- `pipenv install` or `pipenv install --dev` (if we want to reproduce outputs from the jupyter notebook).

### Train the model

To train the model, one needs to download the data from [Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction), save it in `./data/data.csv` and run `python train.py` (inside the virtualenv) or `pipenv run python train.py` (outside the virtualenv).

### Deployment of the web service

To run the web service on a local machine, one needs to execute the following commands:

- ```pipenv run python web_service.py```
- ```pipenv run python request_test.py``` to test that the web service works.

To run the web service on Docker, one needs to execute the following commands:

- ```docker build -t bankruptcy .```
- ```docker run -rm -p 9696:9696 bankruptcy```
- ```pipenv run python request_test.py``` to test that the web service works.

To deploy the web service on AWS Elastic Beanstalk, one needs to create  AWS account and execute the following commands:

- ```pipenv shell```
- ```eb init -p docker -r eu-west-1 bankruptcy```
- ```eb local run --port 9696``` to test if server works locally
- ```eb create bankruptcy-env```. To test that the web service works on the cloud, change `url` in `request_test.py` file to `f'http://{host}/predict`, where `host` variable is the address of the application on the cloud.
- ```eb terminate bankruptcy-env``` to terminate the application.