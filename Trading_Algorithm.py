import numpy as np


def trading_algorithm(x_test, tech_ind_test, scale_back, model):
    buys = []
    sells = []
    thresh = 0.2

    x = 0
    for ohlcv, ind in zip(x_test, tech_ind_test):
        normalised_price_today = ohlcv[-1][0]
        normalised_price_today = np.array([[normalised_price_today]])
        price_today = scale_back.inverse_transform(normalised_price_today)
        predicted = np.squeeze(scale_back.inverse_transform(model.predict([np.array([ohlcv]), np.array([ind])])))
        delta = predicted - price_today
        if delta > thresh:
            buys.append((x, price_today[0][0]))
        elif delta < -thresh:
            sells.append((x, price_today[0][0]))
        x += 1
    return buys, sells
