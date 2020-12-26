from LSTM import lstm_model
from Ploting import plot
from Preprocessing import preprocess

csv_path = 'data/femeli-daily.csv'
history_points = 50
x_train, y_train, x_test, y_test, x_val, y_val, y_test_real, scale_back, tech_ind_train, tech_ind_test, tech_ind_val = preprocess(
    csv_path, history_points)
y_test_predicted = lstm_model(history_points, x_train, y_train, x_test, y_test, x_val, y_val, y_test_real,
                              scale_back, tech_ind_train, tech_ind_test,tech_ind_val)
plot(y_test_real, y_test_predicted)
