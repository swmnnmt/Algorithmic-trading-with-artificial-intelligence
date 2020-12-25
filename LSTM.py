import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
np.random.seed(4)

tf.random.set_seed(4)


def lstm_model(history_points, ohlvc_train, y_train, ohlvc_test, y_test, unscaled_y_test,
               y_normaliser, technical_indicators, tech_ind_train, tech_ind_test):
    # define two sets of inputs
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(33, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch operates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and then output a single value
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)

    adam = optimizers.Adam(lr=0.0005)

    model.compile(optimizer=adam, loss='mse')

    # Fitting model
    from keras.callbacks import ModelCheckpoint
    mcp_save = ModelCheckpoint('./stocks_price.h5', save_best_only=True, monitor='val_loss', mode='min')

    model.fit(x=[ohlvc_train, tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True,
              validation_split=0.38196601125, callbacks=[mcp_save], verbose=0, )
    evaluation = model.evaluate([ohlvc_test, tech_ind_test], y_test)
    print(evaluation)

    y_test_predicted = model.predict([ohlvc_test, tech_ind_test])
    # model.predict returns normalised values
    # now we scale them back up using the y_normaliser from before
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

    # also getting predictions for the entire dataset, just to see how it performs
    y_predicted = model.predict([ohlvc_test, tech_ind_test])
    y_predicted = y_normaliser.inverse_transform(y_predicted)

    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)
    return y_test_predicted