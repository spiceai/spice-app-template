#!/usr/bin/python

"""
Demo Neural Network app
"""

import sys
sys.path[0] = str(sys.path[0]) # BAH!

import os
import json
import dataclasses
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Concatenate, Dense, Dropout, Lambda, LSTM, Reshape

# maybe fix gpu freeze, see: https://stackoverflow.com/a/59297959
configproto = tf.compat.v1.ConfigProto()
configproto.gpu_options.allow_growth = True  # pylint: disable=E1101
sess = tf.compat.v1.Session(config=configproto)
tf.compat.v1.keras.backend.set_session(sess)

@dataclasses.dataclass
class DataSet:
    inputs: np.array
    out: np.array
    scalers: list

def make_tf_data(filled_df: pd.DataFrame, lookback_size: int, lookahead_size) -> Tuple[DataSet, DataSet, pd.DataFrame]:
    window_size = lookback_size + lookahead_size
    lo = filled_df['ts'].min()
    hi = filled_df['ts'].max()
    print(lo, hi, "=", hi - lo)

    sparse = set(filled_df['ts'].to_numpy())
    pad = []
    for i in range(lo, hi):
        if i not in sparse:
            pad.append(i)

    print(len(pad), 'padded')

    pad_df = pd.DataFrame(data={'ts': pad})
    filled_df = pd.concat([filled_df, pad_df])
    filled_df.sort_values('ts', inplace=True, ignore_index=True)

    filled_df = filled_df.interpolate(method='pad') \
        .replace([np.inf, -np.inf], np.nan) \
        .dropna()

    start_points = filled_df['ts'].index.to_numpy()[:-window_size-1].copy()
    np.random.shuffle(start_points)
    print(len(start_points), 'start points')

    split = int(len(start_points) * 0.85)

    def split_window(df, start_indices, window_size, sample=None) -> np.ndarray:
        dataset = []
        if sample and sample <= len(start_indices):
            start_indices = np.random.choice(start_indices, sample)
        for start in start_indices:
            ts = df['y'][start:start+window_size].to_numpy().astype('float32')
            if len(ts) == window_size:
                dataset.append(ts)
        return np.array(dataset)

    raw_train_set = split_window(filled_df, start_points[:split], window_size, sample=1000)
    raw_test_set = split_window(filled_df, start_points[split:], window_size, sample=100)
    print(f'train {len(raw_train_set)}, test {len(raw_test_set)}')

    def create(dataset) -> DataSet:
        data_x, data_y, scalers = [], [], []
        for i in range(0, len(dataset)-window_size, window_size):
            if len(dataset[i:]) < window_size:
                continue

            scaler = StandardScaler()
            back = scaler.fit_transform(dataset[i:i+lookback_size, 0].reshape(-1, 1)).reshape(lookback_size, 1)
            data_x.append(back)
            last = back[-1, 0]
            pred = scaler.transform(dataset[i+lookback_size:i+lookback_size+lookahead_size, 0].reshape(-1, 1)).reshape(lookahead_size, 1)
            data_y.append(pred)
            scalers.append(scaler)
        return DataSet(
            inputs=np.array(data_x),
            out=np.array(data_y),
            scalers=scalers,
        )

    print(f'create train+test sets from raw {raw_train_set.shape} and {raw_test_set.shape}')
    train = create(raw_train_set.reshape(-1, 1))
    test = create(raw_test_set.reshape(-1, 1))
    return (train, test, filled_df)

def make_tf_model(input_shape: Tuple[int, int, int]):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name='inputs')

    x = LSTM(150, return_sequences=True, name='lstm1')(inputs)
    x = Dropout(0.2, name='drop1')(x)

    x = LSTM(150, name='lstm2')(x)
    x = Dropout(0.2, name='drop2')(x)

    dx = Dense(200, name='dense1')(inputs)
    dx = Dropout(0.2, name='ddrop1')(dx)
    dx = Dense(20, name='dense2')(dx)

    dx = Reshape((-1,))(dx)
    x = Reshape((-1,))(x)
    x = Concatenate()([dx, x]) # rm: ix, cx
    x = Dense(30, name='combo')(x)

    output = Dense(1, name='out')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        loss={
            'out': 'mse',
        },
        metrics={
            'out': ['mse', 'mae'],
        },
        optimizer='adam',
    )

    return model

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
    print("RUNNING USER NN TRAINING", context)
    all_data = pd.read_parquet(os.environ['DATA_DIR'])
    print('training data:', all_data.shape)
    all_data.sort_values("ts", inplace=True, ignore_index=True)
    print(all_data.head())

    # TODO(tim) squash the yaml underscore bug so we don't need this kludge
    lookback_size = context.get('lookback_size', context.get('lookbacksize'))
    forecast_size = context.get('forecast_size', context.get('forecastsize'))
    epochs = context.get('epochs')

    train_data, test_data, filled_df = make_tf_data(all_data, lookback_size, forecast_size)
    print(f"train {len(train_data.inputs)}, test {len(test_data.inputs)}")

    model = make_tf_model((None, lookback_size, 1))
    history = model.fit(
        train_data.inputs,
        train_data.out,
        epochs=epochs,
        batch_size=64,
        verbose=1,
    )

    model.save(filepath=os.path.join(os.environ['OUTPUT_DIR'], 'model.keras'), save_format='keras')

    print(dir(history))
    print(dir(history.history))
    report = json.dumps({
      'items': [
        {
          'type': 'html',
          'html': f'''
            <h1>history</h1>
            <div>{json.dumps(history.history)}</div>
          ''',
        }
      ],
    })
    runtime.upload(report, 'report.json')

def infer(context, runtime):
    print('RUNNING USER NN INFERENCE', context)

    model_path = os.path.join(os.environ['MODEL_DIR'], context.get('model_weights_name', context.get('modelweightsname')))
    print('load model from', model_path)
    model = keras.models.load_model(model_path)
    lookback = pd.DataFrame(context['lookback'])
    lookback_size = context.get('lookback_size', context.get('lookbacksize'))
    metadata = context['metadata']

    raw_input_window = lookback['value'][-lookback_size:].to_numpy().reshape(-1).astype("float32")
    scaler = StandardScaler()
    input_window = scaler.fit_transform(raw_input_window.reshape(-1, 1)).reshape(-1, lookback_size)
    print('input shape', input_window.shape)
    predicted = model.predict(input_window)

    now = context['lookback'][-1]['timestamp']
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
    runtime.upload(json.dumps(make_inference_response(predicted, now)), 'results.json')

