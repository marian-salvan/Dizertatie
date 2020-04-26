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

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def plot_predictions(real, predicted, column):
    plt.plot(np.arange(120), real, np.arange(120), predicted)
    plt.legend(('Real temperature', 'Predicted temperature'))
    plt.xlabel('Sample number')
    plt.ylabel('Temperature')
    plt.title(column)
    plt.savefig('./images/one_second_past_inlet_probe_1_rack_1.png')
    plt.show()

    # Create just a figure and only one subplot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')
    ax.set_xlabel('test')
    ax.set_ylabel('test')

    plt.show()
    plt.close(fig)


df = pd.read_csv("./data/data_1_second.csv")

# display all columns
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None

print(df.shape)
print(df.head(5))
# print(df.corr())
print(df.isnull().values.any())

# split data into test and validation using sklearn
feature_col_names = ['inlet_server_past_1', 'inlet_server_past_2',
                     'inlet_server_past_3', 'inlet_server_past_4',
                     't_room_initial', 't_air_input', 'air_flow',
                     'heat_generation_rate_server_1', 'heat_generation_rate_server_2']
predicted_class_names = ['inlet_probe_1_rack_1']

X = df[feature_col_names].values      # predictor feature columns (10 X m)
y = df[predicted_class_names].values  # predicted class column (1 X m)
split_test_size = 0.30

# normalize data
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=split_test_size, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_train.shape)
print(X_test.shape)

model = Sequential()

# input shape = (6,)
model.add(LSTM(9, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(18, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(18))
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear'))

# compile
model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=[r2_keras])

# save model to png
plot_model(model, to_file="./images/one_second_prediction_past.png", show_shapes=True, show_layer_names=True)

# train
BATCH_SIZE = 500
EPOCHS = 5

cbk_early_stopping = EarlyStopping(monitor='val_r2_keras', mode='min')

hist = model.fit(X_train, y_train, BATCH_SIZE, epochs=EPOCHS,
          validation_data=(X_test, y_test),
          callbacks=[cbk_early_stopping])

# save transfer learning model
model.save('first-try.model')
# evaluate the model with the test data to get the scores on "real" data
score = model.evaluate(X_test, y_test, verbose=0)
print("All score", score)

print('Test loss:', score[0])
print('Test mae: ', score[1])

# do predictions
print("Prediction")
predicted_temperature = model.predict(X_test)
predicted_temperature = scaler.inverse_transform(predicted_temperature)
real_temperature = scaler.inverse_transform(y_test)

# plot prediction
plot_predictions(real_temperature[:120], predicted_temperature[:120], 'inlet_probe_1_rack_1')

# plot data to see relationships in training and validation data
epoch_list = list(range(1, len(hist.history['r2_keras']) + 1))  # values for x axis [1, 2, .., # of epochs]
plt.plot(epoch_list, hist.history['r2_keras'], epoch_list, hist.history['val_r2_keras'])
plt.legend(('Training r2_keras', 'Validation r2_keras'))
plt.show()
