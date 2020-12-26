import matplotlib.pyplot as plt


def plot(y_test_real, y_test_predicted):
    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 160
    end = -1

    real = plt.plot(y_test_real[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])

    return plt.show()
