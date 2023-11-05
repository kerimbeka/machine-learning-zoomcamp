import pandas as pd
import xgboost as xgb

from flask import Flask
from flask import request
from flask import jsonify


model = xgb.XGBClassifier()
model.load_model('./model/model.json')

app = Flask('company-bankruptcy-prediction')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = pd.DataFrame.from_dict(client, orient='index').T
    y_pred_proba = model.predict_proba(X)[0, 1]
    y_pred = model.predict(X)

    result = {
        'prediction_probability': round(float(y_pred_proba), 3),
        'prediction': bool(y_pred)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
