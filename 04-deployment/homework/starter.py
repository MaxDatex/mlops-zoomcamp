#!/usr/bin/env python
# coding: utf-8
import sys

import pickle
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

year = int(sys.argv[1])
month = int(sys.argv[2])

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:02d}-{month:02d}.parquet')
print(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:02d}-{month:02d}.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(f'std: {y_pred.std()}')
print(f'mean: {y_pred.mean()}')

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
output_file = 'output/result.parquet'
df_result = pd.DataFrame()

df_result['ride_id'] = df['ride_id']
df_result['pred'] = y_pred


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)




