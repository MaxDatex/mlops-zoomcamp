import os
import pickle
import click

import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    

def init_mlflow(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'
    experiment_name = 'homework-experiment'

    init_mlflow(tracking_uri=MLFLOW_TRACKING_URI, experiment_name=experiment_name)
    mlflow.autolog()
    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        mlflow.set_tag('question', '3')
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)


if __name__ == '__main__':
    run_train()
