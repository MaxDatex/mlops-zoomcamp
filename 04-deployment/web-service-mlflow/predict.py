import pickle
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify


RUN_ID = 'bed12055f0c340ea973ba9607f176cb1'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

# client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer.bin')
# print(f'Downloading dict vectorizer to: {path}')

# with open(path, 'rb') as f_out:
#     dv = pickle.load(f_out)

logged_model = f'runs:/{RUN_ID}/model'

model = mlflow.pyfunc.load_model(logged_model)


# with open('lin_reg.bin', 'rb') as f_in:
#     (dv, model) = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        "duration": pred,
        "model_version": RUN_ID,
    }
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
