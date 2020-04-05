# Losev 7383 var.3 regression target #2

from keras.layers import Input, Embedding, LSTM, Dense, Flatten
from keras.models import Model, Sequential
from pandas import DataFrame as df
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

def loss_plot(label, history):
    plt.clf()
    arg = range(1, len(history['loss']) + 1)
    plt.plot(arg, history['loss'], 'r', label='Training loss')
    plt.plot(arg, history['val_loss'], 'b', label='Validation loss')
    plt.title(label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(label + '.png')
    plt.show()
def build_model(shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
def normalize(data):
    mean = data.mean(axis=0)
    data -= mean
    std = data.std(axis=0)
    data /= std

def generate_dataset(size):
    data = []
    targets = []
    for i in range(size):
        x = np.random.normal(0, 10)
        e = np.random.normal(0, 0.3)
        data.append([x**2 + x + e,
                        math.sin(x - math.pi / 4) + e,
                        math.log(math.fabs(x)) + e,
                        -x**3 + e,
                        -x / 4 + e,
                        -x / 4])
        targets.append(math.fabs(x) + e)
    return np.array(data), np.array(targets)


def build_models():
    encoding_dim = 3
    input = Input(shape=(6,), name='main_input')
    encoded = Dense(encoding_dim, activation='linear')(input)
    encoded_input = Input(shape=(encoding_dim,), name='input_encoded')
    decoded = Dense(6, activation='sigmoid')(encoded_input)
    predicted = Dense(16, activation='relu', kernel_initializer='normal')(encoded)
    predicted = Dense(1, name="out_main")(predicted)

    encoder = Model(input, encoded, name="encoder")
    decoder = Model(encoded_input, decoded, name="decoder")
    predictor = Model(input, predicted, name="regr")
    return encoder, decoder, predictor

data, targets = generate_dataset(500)
datarame = df(np.hstack((data, np.atleast_2d(targets).T))) # 6 columns of data + 1 column of targets
datarame.to_csv("datarame_generated.csv", header=False, index = False)


normalize(data)

model = build_model((6, ))
H = model.fit(data, targets,
                    epochs=50,
                    batch_size=2,
                    verbose=1,
                    validation_split = 0.1)
loss_plot('model with no encoding', H.history)

encoder, decoder, model = build_models()
model.compile(optimizer='rmsprop', loss="mse", metrics=['mae'])
H = model.fit(data, targets,
                 epochs=50,
                 batch_size=2,
                 verbose=1,
                 validation_split = 0.1)
loss_plot('model with encoding', H.history)

# save the data and models
encoded_data = encoder.predict(data)
encoded_datarame = df(encoded_data)
encoded_datarame.to_csv("data_encoded.csv", header=False, index = False)
decoded_datarame = df(decoder.predict(encoded_data))
decoded_datarame.to_csv("data_decoded.csv", header=False, index = False)
decoder.save('decoder.h5')
encoder.save('encoder.h5')
model.save('predictor.h5')