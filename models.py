import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from temperature_softmax import TemperatureSoftmax

img = tf.placeholder(tf.float32, name='img_input', shape=(None, 784))
labels = tf.placeholder(tf.float32, name='labels', shape=(None, 10))


def build_mentor_model_sequential(load=False):
    mentor_model = Sequential()
    # This method of adding the first layer is required to synchronize the
    # Keras and TensorFlow representations
    # See https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
    # for additional information
    l1 = Dense(500, name='mentor_dense_1', activation='sigmoid', input_dim=784)
    l1.set_input(img)
    mentor_model.add(l1)
    mentor_model.add(Dense(10, name='mentor_dense_2'))
    mentor_model.add(TemperatureSoftmax(0.99))
    if load:
        mentor_model.load_weights('mentor.h5')
    return mentor_model


def build_mentee_model_sequential():
    mentee_model = Sequential()
    l1 = Dense(250, name='mentor_dense_1', activation='sigmoid', input_dim=784)
    l1.set_input(img)
    mentee_model.add(l1)
    mentee_model.add(Dense(10, name='mentee_dense_2'))
    mentee_model.add(TemperatureSoftmax(0.99))
    return mentee_model
