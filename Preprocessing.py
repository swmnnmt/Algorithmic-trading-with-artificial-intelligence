import numpy as np
import pandas as pd
from sklearn import preprocessing


def preprocess(csv_path, history_points):
    data = pd.read_csv(csv_path)
    data = data.drop(columns=['date', 'adjClose', 'value', 'count'])
    data = data.drop(0, axis=0)

    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    # using the last {history_points} high low volume close data points, predict the next open price
    ohlvc_histories_normalised = np.array(
        [data_normalised[i: i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array(
        [data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)
    data = data.to_numpy()
    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    assert ohlvc_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]
    return ohlvc_histories_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser
