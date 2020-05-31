import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import keras.callbacks
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys

seq_length = 100
filename = "wonderland.txt"
output_filename = "out.txt"

raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


# saves a seed and the text generated to the file specified
def generate(model):
    # pick a random seed
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    seed = [int_to_char[value] for value in pattern]
    f = open(output_filename, 'a')
    f.write("\nSeed:" + "\"" + ''.join(seed)+ "\"\n")
    f.write("prediction: ")
    # generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        f.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    f.close()

class CustomCB(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        f = open(output_filename, 'a')
        f.write("\nEpoch: "+str(epoch)+"\n")
        f.close()
        generate(model)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, CustomCB()]

model.fit(X[0:20], y[0:20], epochs=20, batch_size=128, callbacks=callbacks_list)
generate(model)
model.save('model.h5')