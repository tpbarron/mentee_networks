import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense

def build_mentor_model(load=False):
    mentor_model = Sequential()
    mentor_model.add(Dense(500, activation='sigmoid', input_dim=784))
    mentor_model.add(Dense(10, activation='softmax'))
    #mentor_preds = mentor_model(img)
    if load:
        mentor_model.load_weights('mentor.h5')
    return mentor_model


def build_mentee_model():
    mentee_model = Sequential()
    mentee_model.add(Dense(500, activation='sigmoid', input_dim=784))
    mentee_model.add(Dense(10, activation='softmax'))
    #mentee_preds = mentee_model(img)
    return mentee_model
