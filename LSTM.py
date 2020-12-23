import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras.models import Model

np.random.seed(4)
tf.random.set_seed(4)


def lstm_model(history_points, ohlvc_train, y_train, ohlvc_test, y_test, ohlvc_histories, unscaled_y_test, y_normaliser):
    # Creating LSTM model
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    x = LSTM(22, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')

    # Fitting model
    mcp_save = ModelCheckpoint('./stocks_price.h5', save_best_only=True, monitor='val_loss', mode='min')
    model.fit(x=ohlvc_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1,
              callbacks=[mcp_save], verbose=0)
    evaluation = model.evaluate(ohlvc_test, y_test)
    print(evaluation)
    y_test_predicted = model.predict(ohlvc_test)
    # model.predict returns normalised values
    # now we scale them back up using the y_normaliser
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    # also getting predictions for the entire dataset, just to see how it performs
    y_predicted = model.predict(ohlvc_histories)
    y_predicted = y_normaliser.inverse_transform(y_predicted)

    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)
