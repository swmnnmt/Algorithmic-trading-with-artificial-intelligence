import matplotlib.pyplot as plt


def plot(unscaled_y_test, y_test_predicted):
    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 160
    end = -1

    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])

    return plt.show()
