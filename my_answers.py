import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    """
    Runs a sliding window along the input series and creates associated input/output pairs.
    """
    # containers for input/output pairs
    X = []
    y = []

    start = 0
    for num in series:
        if window_size <= len(series) - 1:
            X.append(series[start:window_size])
            y.append(series[window_size])
            start += 1
            window_size += 1

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    """
    A two hidden layer RNN model of the following specification:
    - Layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    - Layer 2 uses a fully connected module with one unit
    """
    model = Sequential()
    model.add(LSTM(5, input_shape = (window_size, 1)))
    model.add(Dense(1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    """
    Clean the text by removing all of the atypical characters.
    """
    punctuation = ['!', ',', '.', ':', ';', '?']
    replace_list = set()
    for character in text:
        if character.islower() == True:
            pass
        elif (character in punctuation) == True:
            pass
        elif character == " ":
            pass
        else:
            replace_list.add(character)
    for character in replace_list:
        text = text.replace(character, " ")
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    """
    Runs a sliding window along the input text and creates associated input/output pairs.
    """
    # containers for input/output pairs
    inputs = []
    outputs = []

    start = 0
    for num in text:
        if window_size <= len(text) - 1:
            inputs.append(text[start:window_size])
            outputs.append(text[window_size])
            start += step_size
            window_size += step_size

    return inputs,outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    """
    A 3 layer RNN model of the following specification:
    - layer 1 uses an LSTM module with 200 hidden units
    - layer 2 uses a linear module, fully connected, with len(chars) hidden units
    - layer 3 uses a softmax activation
    """
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
