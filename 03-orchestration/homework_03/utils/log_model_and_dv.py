import os
import pickle
from typing import Dict, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from mlflow import MlflowClient
from mlflow.data import from_numpy, from_pandas
from mlflow.entities import DatasetInput, InputTag, Run
from mlflow.models import infer_signature, signature
from mlflow.sklearn import log_model as log_model_sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer

DEFAULT_DEVELOPER = os.getenv('EXPERIMENTS_DEVELOPER', 'Maksym')
DEFAULT_EXPERIMENT_NAME = 'nyc-taxi-experiment-homework'
DEFAULT_TRACKING_URI = 'http://mlflow:5000'


def setup_experiment(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> Tuple[MlflowClient, str]:
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = client.create_experiment(experiment_name)

    return client, experiment_id


def track_experiment(
    experiment_name: Optional[str] = None,
    developer: Optional[str] = None,
    model: Optional[Union[BaseEstimator, xgb.Booster]] = None,
    dv: Optional[DictVectorizer] = None,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    verbosity: Union[
        bool, int
    ] = False,  # False by default or else it creates too many logs
    **kwargs,
) -> Run:
    experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
    tracking_uri = tracking_uri or DEFAULT_TRACKING_URI

    print(os.listdir())

    client, experiment_id = setup_experiment(experiment_name, tracking_uri)

    run = client.create_run(experiment_id, run_name=run_name or None)
    run_id = run.info.run_id

    for key, value in [
        ('developer', developer or DEFAULT_DEVELOPER),
        ('model', model.__class__.__name__),
    ]:
        if value is not None:
            client.set_tag(run_id, key, value)

    if model:
        log_model = None

        if isinstance(model, BaseEstimator):
            log_model = log_model_sklearn

        if log_model:
            opts = dict(artifact_path='models', input_example=None)

            log_model(model, **opts)
            if verbosity:
                print(f'Logged model {model.__class__.__name__}.')

    if dv:
        with open("/home/mlops/homework_03/models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
            
        mlflow.log_artifact("/home/mlops/homework_03/models/preprocessor.b", artifact_path="preprocessor")

    return run