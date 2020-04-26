import pandas as pd  # pandas is a dataframe library
import numpy as np
import matplotlib.pyplot as plt  # matplotlib.pyplot plots data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras import backend as K
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

OUTPUTS = ['inlet_probe_1_rack_1', 'inlet_probe_2_rack_1', 'inlet_probe_3_rack_1', 'inlet_probe_4_rack_1',
           'inlet_probe_1_rack_2', 'inlet_probe_2_rack_2', 'inlet_probe_3_rack_2', 'inlet_probe_4_rack_2',
           'outlet_probe_1_rack_1', 'outlet_probe_1_rack_2', 'outlet_probe_1_room',
           'outlet_probe_2_room', 'outlet_probe_3_room', 'outlet_probe_4_room']


def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def plot_predictions(real, predicted, column):
    fig, ax = plt.subplots()
    line1, = ax.plot(np.arange(60), real)
    line2, = ax.plot(np.arange(60), predicted)
    line1.set_label('Real temperature')
    line2.set_label('Predicted temperature')
    ax.set_title(column)
    ax.set_xlabel('Sample number')
    ax.set_ylabel('Temperature')
    ax.legend()
    plt.savefig('./images/one_second_data/' + column + '.png')
    plt.close(fig)


for column in OUTPUTS:
    df = pd.read_csv("./data/data.log")

    feature_col_names = ['t_room_initial', 't_air_input', 'air_flow', 'heat_generation_rate_server_1',
                         'heat_generation_rate_server_2', 'elapsed_time']

    X = df[feature_col_names].values  # predictor feature columns (10 X m)
    y = df[[column]].values  # predicted class column (1 X m)
    split_test_size = 0.30

    # normalize data
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=split_test_size, random_state=42)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()

    # input shape = (6,)
    model.add(LSTM(6, input_shape=(X_train.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(20, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(20))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='linear'))

    # compile
    model.compile(loss='mean_absolute_error',
                  optimizer='adam',
                  metrics=[r2_keras])

    # save model to png
    plot_model(model, to_file="./images/one_second_prediction.png", show_shapes=True, show_layer_names=True)

    # train
    BATCH_SIZE = 500
    EPOCHS = 5

    cbk_early_stopping = EarlyStopping(monitor='val_r2_keras', mode='max')

    hist = model.fit(X_train, y_train, BATCH_SIZE, epochs=EPOCHS,
                     validation_data=(X_test, y_test),
                     callbacks=[cbk_early_stopping])

    # evaluate the model with the test data to get the scores on "real" data
    score = model.evaluate(X_test, y_test, verbose=2)
    print("All score", score)

    print('Test loss:', score[0])
    print('Test r2: ', score[1])

    file = open("./results/one_second_data/result.txt", "a")
    file.write("Column: " + column + "\n")
    file.write("Test loss, Test r2" + "\n")
    file.write(str(score[0]) + " " + str(score[1]) + "\n")
    file.write("\n")
    file.close()

    # do predictions
    print("Prediction")
    predicted_temperature = model.predict(X_test)
    predicted_temperature = scaler.inverse_transform(predicted_temperature)
    real_temperature = scaler.inverse_transform(y_test)

    # plot prediction
    plot_predictions(real_temperature[:60], predicted_temperature[:60], column)

    # plot data to see relationships in training and validation data
    # epoch_list = list(range(1, len(hist.history['r2_keras']) + 1))  # values for x axis [1, 2, .., # of epochs]
    # plt2.plot(epoch_list, hist.history['r2_keras'], epoch_list, hist.history['val_r2_keras'])
    # plt2.legend(('Training r2_keras', 'Validation r2_keras'))
    # plt2.savefig('./images/one_second_data/' + column + '_learning.png')
    # plt.show()
