from LSTM import lstm_model
from Ploting import plot
from Preprocessing import preprocess
from Trading_Algorithm import trading_algorithm, compute_earnings

csv_path = 'data/tickers_data/femeli-daily.csv'
history_points = 50

x_train, y_train, x_test, y_test, x_val, y_val, y_test_real, scale_back, tech_ind_train, tech_ind_test, tech_ind_val = preprocess(
    csv_path, history_points)
y_test_predicted, model = lstm_model(history_points, x_train, y_train, x_test, y_test, x_val, y_val, y_test_real,
                                     scale_back, tech_ind_train, tech_ind_test, tech_ind_val)
buys, sells = trading_algorithm(x_test, tech_ind_test, scale_back, model)
plot(y_test_real, y_test_predicted, buys, sells)
purchase_amt = 1000000000  # 100 million Toman
print("{} Rials after trading over trading days of the test data will make profit {} Rials".format(purchase_amt,
                                                                                                   compute_earnings(
                                                                                                       buys,
                                                                                                       sells,
                                                                                                       purchase_amt) - purchase_amt))
