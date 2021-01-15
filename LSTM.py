import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error

np.random.seed(4)

tf.random.set_seed(4)


def lstm_model(history_points, x_train, y_train, x_test, y_test, x_val, y_val, y_test_real, scale_back,
               tech_ind_train, tech_ind_test, tech_ind_val):
    # define two sets of inputs
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    dense_input = Input(shape=(tech_ind_train.shape[1],), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
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

    model.compile(optimizer=adam, loss=MeanSquaredError())

    # Fitting model
    mcp_save = ModelCheckpoint('./stocks_price.h5', save_best_only=True, monitor='val_loss', mode='min')

    model.fit(x=[x_train, tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True,
              validation_data=([x_val, tech_ind_val], y_val), callbacks=[mcp_save], verbose=0)
    model.load_weights('./stocks_price.h5')

    # Evaluate model (scaled data)
    evaluation = model.evaluate([x_test, tech_ind_test], y_test)
    print("Prediction Error for normalized data : {}".format(evaluation))

    # Evaluate model (unscaled data)
    y_test_predicted = model.predict([x_test, tech_ind_test])
    y_test_predicted = scale_back.inverse_transform(y_test_predicted)
    real_rmse = mean_squared_error(y_test_real,y_test_predicted,squared=False)

    print("Adjusted Prediction Mean Squared Error for real data : {}  ".format(real_rmse))

    return y_test_predicted, model
