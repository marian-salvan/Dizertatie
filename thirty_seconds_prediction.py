import pandas as pd                 # pandas is a dataframe library
import numpy as np
import matplotlib.pyplot as plt      # matplotlib.pyplot plots data
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
    return (1 - SS_res/(SS_tot + K.epsilon()))

def plot_predictions(real, predicted, column):
    fig, ax = plt.subplots()
    line1, = ax.plot(np.arange(30), real)
    line2, = ax.plot(np.arange(30), predicted)
    line1.set_label('Real temperature')
    line2.set_label('Predicted temperature')
    ax.set_title(column)
    ax.set_xlabel('Second')
    ax.set_ylabel('Temperature')
    ax.legend()
    plt.savefig('./images/thirty_seconds_data/' + column + '.png')
    plt.close(fig)


def plot_learning_rates(hist, column, score):
    epoch_list = list(range(1, len(hist.history[score]) + 1))  # values for x axis [1, 2, .., # of epochs]

    fig, ax = plt.subplots()
    line1, = ax.plot(epoch_list, hist.history[score])
    line2, = ax.plot(epoch_list, hist.history['val_' + score])
    line1.set_label('Training ' + score)
    line2.set_label('Validation ' + score)
    ax.set_title(column)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.legend()
    plt.savefig('./images/thirty_seconds_data/' + score + '_' + column + '.png')
    plt.close(fig)


prediction_index = 0

for column in OUTPUTS:

    df = pd.read_csv("E:/data/thirty_seconds_data/" + column + ".csv")
    print(df.shape)

    X = df[df.columns[0:65]].values      # predictor feature columns (10 X m)
    y = df[df.columns[65:95]].values  # predicted class column (1 X m)
    split_test_size = 0.2

    # normalize data
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=split_test_size, random_state=42)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()

    # input shape = (6,)
    model.add(LSTM(65, input_shape=(X_train.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(130, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(130))
    model.add(Dropout(0.2))

    model.add(Dense(30, activation='linear'))

    # compile
    model.compile(loss='mean_absolute_error',
                  optimizer='adam',
                  metrics=[r2_keras])

    # save model to png
    plot_model(model, to_file="./images/thirty_seconds_data.png", show_shapes=True, show_layer_names=True)

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

    # do predictions
    print("Prediction")
    predicted_temperature = model.predict(X_test)
    predicted_temperature = scaler.inverse_transform(predicted_temperature)
    real_temperature = scaler.inverse_transform(y_test)

    # plot prediction
    print('Predicted: ', predicted_temperature[0])
    print('Real: ', real_temperature[0])

    file = open("./results/thirty_seconds_data/result.txt", "a")
    file.write("Column: " + column + "\n")
    file.write("Test loss, Test r2" + "\n")
    file.write("Test loss, Test r2" + "\n")
    file.write(str(score[0]) + " " + str(score[1]) + "\n")
    file.write("\n")
    file.close()

    plot_predictions(real_temperature[prediction_index], predicted_temperature[prediction_index], column)

    prediction_index = prediction_index + 1

    plot_learning_rates(hist, column, 'r2_keras')
    plot_learning_rates(hist, column, 'loss')
