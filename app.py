#!/usr/bin/env python

"""
Author: Srdjan Santic
Date: 18-Feb-2020

This script runs a simple Flask server, listening by default on port 5000.
It requires a pickled model, receives requests in JSON format, and outputs
predictions.

The correct usage of the script is:

$ python model.pickle

"""


import argparse
import pickle
import numpy as np
import pickle
from flask import Flask, request, jsonify

parser = argparse.ArgumentParser(description='Script for running a Flask server.')
parser.add_argument('model', type=str, help='Input pickled model for scoring.')
args = parser.parse_args()

app = Flask(__name__)


@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    prediction = np.array2string(model.predict(data))

    return jsonify(prediction)


if __name__ == '__main__':
    modelfile = args.model
    model = pickle.load(open(modelfile, 'rb'))
    app.run(debug=True, host='0.0.0.0')
