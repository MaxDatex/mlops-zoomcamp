#!/usr/bin/env python
# coding: utf-8
import sys

import pickle
import pandas as pd
from flask import Flask, request, jsonify


def get_model(path):
    with open(path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def read_data(filename):
    df = pd.read_parquet(filename)

    categorical = ['PULocationID', 'DOLocationID']
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def get_prediction(df, dv, model):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred


app = Flask('predict-duration-mean')


@app.route('predict/', methods=['POST'])
def predict():
    data = request.get_json()
    year = data['year']
    month = data['month']
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:02d}-{month:02d}.parquet')
    pred = get_prediction(df, get_model('model.bin'))
    result = {
        "mean": pred.mean(),
        "std": pred.std()
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    