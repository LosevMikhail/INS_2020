import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing

def draw_plot(label, train, val, blocks_num):
    plt.clf()
    arg = range(1, len(train) + 1)
    plt.plot(arg, train, 'r', label='Training')
    plt.plot(arg, val, 'b', label='Validation')
    plt.title('Training and validation ' + label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(str(blocks_num) + '_MAE_' + label + '.png')
#     plt.show()

def normalize(data):
    mean = data.mean(axis=0)
    data -= mean
    std = data.std(axis=0)
    data /= std

def build_model(shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def fold(data, targets, k, num_epochs = 100):
    num_val_samples = len(data) // k
    fold_history = {'mae': [], 'val_mae': []}
    for i in range(k):
        print('processing fold #', i)
        val_data = data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = targets[i * num_val_samples: (i + 1) * num_val_samples]
        train_data = np.concatenate(
            [data[:i * num_val_samples], data[(i + 1) * num_val_samples:]], axis=0)
        train_targets = np.concatenate(
            [targets[:i * num_val_samples], targets[(i + 1) * num_val_samples:]], axis=0)
        model = build_model((train_data.shape[1],))
        H = model.fit(train_data, train_targets, epochs=num_epochs, batch_size=1, verbose=0,
            validation_data=(val_data, val_targets))
        mae = H.history['mean_absolute_error']
        val_mae = H.history['val_mean_absolute_error']
        loss = H.history['loss']
        val_loss = H.history['val_loss']
        fold_history['mae'].append(mae)
        fold_history['val_mae'].append(val_mae)
        draw_plot('MAE (block #' + str(i) + ')', mae, val_mae, k)
    draw_plot('fold MAE',
        [np.mean([x[i] for x in fold_history['mae']]) for i in range(num_epochs)],
        [np.mean([x[i] for x in fold_history['val_mae']]) for i in range(num_epochs)], k)


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# All the data available is both train and test, default separation is conditional
data = np.concatenate([train_data, test_data], axis=0)
targets = np.concatenate([train_targets, test_targets], axis=0)
normalize(data)
fold(data, targets, 10)
