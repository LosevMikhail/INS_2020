import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
import os
import string


picdir = './pics/'

def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def getModel(dimension = 10000):
    model = models.Sequential()
    # Input - Layer
    model.add(layers.Dense(50, activation = "relu", input_shape=(dimension, )))
    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    # Output- Layer
    model.add(layers.Dense(1, activation = "sigmoid"))
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return model

def loadData(num_words=10000):
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=num_words)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    return data, targets

def savePlot(label, history):
    plt.clf()
    arg = range(1, len(history['accuracy']) + 1)
    plt.plot(arg, history['accuracy'], 'r', label='Training acc')
    plt.plot(arg, history['val_accuracy'], 'b', label='Validation acc')
    plt.title(label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(label + '.png')
    plt.savefig(picdir + label + '.png')

def exploreDimensionValue(dim):
    model = getModel(dim)
    data, targets = loadData(dim)
    data = vectorize(data, dim)
    targets = np.array(targets).astype("float32")
    results = model.fit(data, targets, epochs= 50, batch_size = 500, validation_split=0.1)
    savePlot(str(dim) + " dimensions", results.history)
    return results.history['val_accuracy'][-1]

def encode(text):
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    text = text.lower() # to lowercase cuz there's no indexes for non-lowercase words
    words = text.split(" ")
    index = imdb.get_word_index()
    encoded = [index.get(word, 0) for word in words]
    return encoded

def makePrediction(review, model, dim = 10000):
    vectorized = vectorize([encode("Hello, little dear!")], dim)
    return model.predict(vectorized)[0][0]


# Collect data
dimAcc = dict()
for dim in [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]:
    acc = exploreDimensionValue(dim)
    dimAcc[dim] = acc
# Draw a diagram
fig, ax = plt.subplots()
plt.ylabel('Accuracy')
plt.xlabel('Dimensions')
ax.bar([str(key) for key in dimAcc.keys()], dimAcc.values())
ax.set_facecolor('seashell')
fig.set_facecolor('floralwhite')
plt.savefig(picdir + 'diagram.png')


# Make a prediction for user review
review = """
        # this film was just brilliant casting location scenery story direction everyone's really suited the part they played
        and you could just imagine being there robert # is an amazing actor and now the same being director # father came from
        the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks
        throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for #
        and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and
        you know what they say if you cry at a film it must have been good and this definitely was also # to the two little
        boy's that played the # of norman and paul they were just brilliant children are often left out of the # list i think
        because the stars that play them all grown up are such a big profile for the whole film but these children are amazing
        and should be praised for what they have done don't you think the whole story was so lovely because it was true and was
        someone's life after all that was shared with us all
    """
# Get a model
dim = 10000
model = getModel(dim)
data, targets = loadData(dim)
data = vectorize(data, dim)
targets = np.array(targets).astype("float32")
model.fit(data, targets, epochs= 50, batch_size = 500, validation_split=0.1)
# '1 - prediction' cuz good reviews label is 1, bad ones label is 0
print('It is ' + str(1 - makePrediction(review, model, dim)) + ' percent good')

# Wake up Neo
os.system('play --no-show-progress --null --channels 1 synth 1 sine 440')