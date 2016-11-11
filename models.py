import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense


def build_mentor_model(img, load=False):
    # Keras layers can be called on TensorFlow tensors:
    l1 = Dense(128, activation='sigmoid')(img)  # fully-connected layer with 128 units and ReLU activation
    preds = Dense(10, activation='softmax')(l1)
    # if load:
    #     preds.load_weights('mentor.h5')
    return [l1, preds]

def build_mentee_model(img):
    # Keras layers can be called on TensorFlow tensors:
    l1 = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
    preds = Dense(10, activation='softmax')(l1)
    return [l1, preds]


def build_mentor_model_sequential(load=False):
    mentor_model = Sequential()
    mentor_model.add(Dense(500, activation='sigmoid', input_dim=784))
    mentor_model.add(Dense(10, activation='softmax'))
    if load:
        mentor_model.load_weights('mentor.h5')
    return mentor_model


def build_mentee_model_sequential():
    mentee_model = Sequential()
    mentee_model.add(Dense(500, activation='sigmoid', input_dim=784))
    mentee_model.add(Dense(10, activation='softmax'))
    #mentee_preds = mentee_model(img)
    return mentee_model
