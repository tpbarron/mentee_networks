import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from temperature_softmax import TemperatureSoftmax

img = tf.placeholder(tf.float32, name='img_input', shape=(None, 784))
labels = tf.placeholder(tf.float32, name='labels', shape=(None, 10))

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
    mentor_model.add(Dense(500, name='mentor_dense_1', activation='sigmoid', input_dim=784))
    mentor_model.add(Dense(10, name='mentor_dense_2')) #, activation='sigmoid'))
    mentor_model.add(TemperatureSoftmax(0.99))
    if load:
        mentor_model.load_weights('mentor.h5')
    return mentor_model


def build_mentee_model_sequential():
    mentee_model = Sequential()
    mentee_model.add(Dense(500, name='mentee_dense_1', activation='sigmoid', input_dim=784))
    mentee_model.add(Dense(10, name='mentee_dense_2')) #, activation='sigmoid'))
    mentee_model.add(TemperatureSoftmax(0.99))
    return mentee_model
