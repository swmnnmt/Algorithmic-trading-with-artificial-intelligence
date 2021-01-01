import matplotlib.pyplot as plt


def plot(y_test_real, y_test_predicted, buys, sells):
    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 0
    end = -1

    plt.plot(y_test_real[start:end], label='real')
    plt.plot(y_test_predicted[start:end], label='predicted')

    plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00', label='Buy')
    plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000', label='Sell')

    plt.legend(['Real', 'Predicted', 'Buy', 'Sell'])

    return plt.show()
