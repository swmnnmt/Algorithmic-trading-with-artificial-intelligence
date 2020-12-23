from LSTM import lstm_model
from Preprocessing import preprocess

csv_path = 'femeli-daily.csv'
history_points = 22  # 1 working month in Tehran Stock Market
ohlvc_histories, next_day_open_values, unscaled_y, y_normaliser = preprocess(csv_path, history_points)
split_ratio = 1.61803398875  # the Fibonacci golden ratio of data to be used for train and test
n = int(ohlvc_histories.shape[0] / split_ratio)
# splitting the dataset up into train and test sets
ohlvc_train = ohlvc_histories[:n]
y_train = next_day_open_values[:n]
ohlvc_test = ohlvc_histories[n:]
y_test = next_day_open_values[n:]
unscaled_y_test = unscaled_y[n:]

lstm_model(history_points, ohlvc_train, y_train, ohlvc_test, y_test, ohlvc_histories, unscaled_y_test, y_normaliser)
