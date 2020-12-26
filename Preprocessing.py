import numpy as np
import pandas as pd
from sklearn import preprocessing


def preprocess(csv_path, history_points):
    data = pd.read_csv(csv_path)
    data = data.drop(columns=['date', 'adjClose', 'value', 'count'])
    data = data.drop(0, axis=0)
    data_np = data.to_numpy()
    """
    Train-Test-Validation Split
    """
    n1 = int(data_np.shape[0] * 0.7)
    n2 = int((data_np.shape[0] - n1) / 2)
    x_train = data_np[:n1]
    x_val = data_np[n1: n1 + n2]
    x_test = data_np[n1 + n2:]
    y_train_real = slicing(x_train, history_points)[1]
    y_train_real = np.expand_dims(y_train_real, -1)
    scale_back = preprocessing.MinMaxScaler()
    scale_back.fit(y_train_real)
    y_test_real = slicing(x_test, history_points)[1]

    minmax_scale = preprocessing.MinMaxScaler().fit(x_train)
    x_train_n = minmax_scale.transform(x_train)
    x_val_n = minmax_scale.transform(x_val)
    x_test_n = minmax_scale.transform(x_test)

    ohlvc_train, y_train = slicing(x_train_n, history_points)
    x_val_n, y_val = slicing(x_val_n, history_points)
    ohlvc_test, y_test = slicing(x_test_n, history_points)
    y_train = np.expand_dims(y_train, -1)

    """
    SMA Technical Indicator
    """
    x_train_ind = slicing(x_train, history_points)[0]
    tech_ind_train = []
    for his in x_train_ind:
        # since we are using his[1,2,4] we are taking the SMA of the high, low ,closing price
        sma = np.mean(his[:, [1, 2, 4]])
        tech_ind_train.append(np.array([sma]))

    x_test_ind = slicing(x_test, history_points)[0]
    tech_ind_test = []
    for his in x_test_ind:
        # since we are using his[1,2,4] we are taking the SMA of the high, low ,closing price
        sma = np.mean(his[:, [1, 2, 4]])
        tech_ind_test.append(np.array([sma]))

    x_val_ind = slicing(x_val, history_points)[0]
    tech_ind_val = []
    for his in x_val_ind:
        # since we are using his[1,2,4] we are taking the SMA of the high, low ,closing price
        sma = np.mean(his[:, [1, 2, 4]])
        tech_ind_val.append(np.array([sma]))

    tech_ind_scaler = preprocessing.MinMaxScaler().fit(tech_ind_train)
    tech_ind_train = tech_ind_scaler.transform(tech_ind_train)
    tech_ind_test = tech_ind_scaler.transform(tech_ind_test)
    tech_ind_val = tech_ind_scaler.transform(tech_ind_val)

    tech_ind_train = np.array(tech_ind_train)
    tech_ind_test = np.array(tech_ind_test)
    tech_ind_val = np.array(tech_ind_val)

    assert ohlvc_train.shape[0] == y_train.shape[0] == tech_ind_train.shape[0]
    return ohlvc_train, y_train, ohlvc_test, y_test, x_val_n, y_val, y_test_real, scale_back, tech_ind_train, tech_ind_test, tech_ind_val


def slicing(data, history_points):
    # using the last {history_points} open high low close volume data points, predict the next open value
    ohlvc_histories = np.array(
        [data[i: i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.array(
        [data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])

    return ohlvc_histories, next_day_open_values
