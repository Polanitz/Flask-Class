import pickle
from flask import Flask, request, jsonify, render_template
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return "Hi, this is the home page of the app!!! v0.4"

# @app.route('/stam')
# def stam():
#     return "stam stam 0.1"

@app.route('/predict', methods=['POST'])
def predict():
    print(request)
    data = request.get_json(force=True)
    print("Data:", data)
    df = pd.DataFrame(np.nan, index=[0], columns=data.keys())
    df.loc[0, data.keys()] = data.values()
    pred = model.predict(df)
    print("PRED:", pred)
    prob = model.predict_proba(df)
    print("PROBA:", prob)
    d = dict()
    d['pred'] = pred[0]
    d['proba'] = prob[0][0]
    print("aa")
    return jsonify(d)


if __name__ == "__main__":
    with open("my_model.pickle", 'rb') as file:
        model = pickle.load(file)
        print("load is ok!!!")
    print("new line")
    app.run(debug=True, port=5805)
