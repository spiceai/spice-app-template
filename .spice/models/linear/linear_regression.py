#!/usr/bin/python

"""
Demo Linear Regression app
"""

import os
import math
import json
import dataclasses
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

def make_inference_response(predicted: np.ndarray, now: float):
    forecast = []
    for npvalue in predicted:
        # block based model, so time is dense, no "12 sec" timestamp extrapolation required
        now += 1
        value = float(npvalue)
        if math.isnan(value):
            value = -1e10
        forecast.append({
          'timestamp': now,
          'value': value,
        })

    return {
        'forecast': forecast,
    }

def train(context, runtime):
    runtime.upload('', 'model.txt')
    report = json.dumps({
      'items': [
        {
          'type': 'html',
          'html': '<div>N/A</div>',
        }
      ],
    })
    runtime.upload(report, 'report.json')

def infer(context, runtime):
    print('RUNNING USER LINEAR REGRESSION INFERENCE', context)

    lookback = pd.DataFrame(context['lookback'])
    lookback_size = context.get('lookback_size', context.get('lookbacksize'))
    forecast_size = context.get('forecast_size', context.get('forecastsize'))
    metadata = context['metadata']

    X = lookback['timestamp'][-lookback_size:].to_numpy().reshape(-1, 1).astype("float32")
    y = lookback['value'][-lookback_size:].to_numpy().reshape(-1).astype("float32")
    reg = LinearRegression().fit(X, y)
    print('score', reg.score(X, y))

    now = context['lookback'][-1]['timestamp']
    predicted = reg.predict(np.arange(now + 1, now + 1 + forecast_size).reshape(-1, 1))
    runtime.upload(json.dumps(make_inference_response(predicted, now)), 'results.json')

