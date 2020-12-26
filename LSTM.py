import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras.models import Model

np.random.seed(4)

tf.random.set_seed(4)


def lstm_model(history_points, ohlvc_train, y_train, ohlvc_test, y_test, x_val_n, y_val, y_test_real, scale_back):
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')

    # Fitting model
    from keras.callbacks import ModelCheckpoint
    mcp_save = ModelCheckpoint('./stocks_price.h5', save_best_only=True, monitor='val_loss', mode='min')

    model.fit(x=ohlvc_train, y=y_train, batch_size=32, epochs=50, shuffle=True,
              validation_data=(x_val_n, y_val), callbacks=[mcp_save], verbose=0)
    model.load_weights('./stocks_price.h5')
    evaluation = model.evaluate(ohlvc_test, y_test)
    print("Prediction Error for normalized data : {}".format(evaluation))

    y_test_predicted = model.predict(ohlvc_test)
    y_test_predicted = scale_back.inverse_transform(y_test_predicted)

    real_mse = np.mean(np.square(y_test_real - y_test_predicted))
    scaled_mse = real_mse / (np.max(y_test_real) - np.min(y_test_real)) * 100
    print("Prediction Error for real data : {}".format(scaled_mse))

    return y_test_predicted
