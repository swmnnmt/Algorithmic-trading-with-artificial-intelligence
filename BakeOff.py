from LSTM import lstm_model
from Ploting import plot
from Preprocessing import preprocess

csv_path = 'data/femeli-daily.csv'
history_points = 50
ohlvc_train, y_train, ohlvc_test, y_test, x_val_n, y_val, y_test_real, scale_back = preprocess(csv_path, history_points)
y_test_predicted = lstm_model(history_points, ohlvc_train, y_train, ohlvc_test, y_test, x_val_n, y_val, y_test_real,
                              scale_back)
plot(y_test_real, y_test_predicted)
