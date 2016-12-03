import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Reshape, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

################################################################################
# Model creation
################################################################################

img_dense = tf.placeholder(tf.float32, name='img_input', shape=(None, 784))
img_conv = tf.placeholder(tf.float32, name='img_input', shape=(None, 28, 28, 1))
img_cifar = tf.placeholder(tf.float32, name='img_input', shape=(None, 32, 32, 3))
labels = tf.placeholder(tf.float32, name='labels', shape=(None, 10))


################################################################################
# Dense MNIST models
################################################################################

def build_mentor_model(load=False):
    mentor_model = Sequential()
    # This method of adding the first layer is required to synchronize the
    # Keras and TensorFlow representations
    # See https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
    # for additional information
    l1 = Dense(500, name='mentor_dense_1', activation='sigmoid', input_dim=784)
    l1.set_input(img_dense)
    mentor_model.add(l1)
    mentor_model.add(Dense(250, activation='sigmoid'))
    mentor_model.add(Dense(10, name='mentor_dense_2'))
    mentor_model.add(Activation('softmax'))
    if load:
        print ("Loading saved model")
        mentor_model.load_weights('mentor_dense.h5')
    return mentor_model


def build_mentee_model():
    mentee_model = Sequential()
    l1 = Dense(300, name='mentor_dense_1', activation='sigmoid', input_dim=784)
    l1.set_input(img_dense)
    mentee_model.add(l1)
    mentee_model.add(Dense(150, activation='sigmoid'))
    mentee_model.add(Dense(10, name='mentee_dense_2'))
    mentee_model.add(Activation('softmax'))
    return mentee_model


################################################################################
# Convolutional MNIST models
################################################################################

def build_mentor_model_conv(load=False):
    mentor_model = Sequential()
    # This method of adding the first layer is required to synchronize the
    # Keras and TensorFlow representations
    # See https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
    # for additional information
    l1 = Convolution2D(20, 5, 5, subsample=(1, 1), border_mode='same', activation='relu', input_shape=(28, 28, 1))
    l1.set_input(img_conv)
    mentor_model.add(l1)
    mentor_model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    mentor_model.add(Convolution2D(50, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    mentor_model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    mentor_model.add(Flatten())
    mentor_model.add(Dense(64, activation='relu'))
    mentor_model.add(Dense(64, activation='relu'))
    mentor_model.add(Dense(10))
    mentor_model.add(Activation('softmax'))
    if load:
        print ("Loading saved model")
        mentor_model.load_weights('models/mentor_conv_mnist.h5')
    return mentor_model


def build_mentee_model_conv():
    mentee_model = Sequential()
    l1 = Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='same', activation='relu', input_shape=(28, 28, 1))
    l1.set_input(img_conv)
    mentee_model.add(l1)
    mentee_model.add(Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='same', activation='relu'))
    mentee_model.add(Flatten())
    mentee_model.add(Dense(10, name='mentee_dense_2'))
    mentee_model.add(Activation('softmax'))
    return mentee_model


################################################################################
# DQN models
################################################################################

def build_mentor_model_dqn(inputs, num_actions, load=False):
    inputs = tf.transpose(inputs, [0, 2, 3, 1])
    dqn = Sequential()
    l1 = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(84, 84, 4))
    l1.set_input(inputs)
    dqn.add(l1)
    dqn.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
    dqn.add(Flatten())
    dqn.add(Dense(256, activation='relu'))
    dqn.add(Dense(num_actions))
    if load:
        print ("Loading saved model")
        dqn.load_weights('mentor_dqn.h5')
    return dqn


def build_mentee_model_dqn(inputs, num_actions):
    inputs = tf.transpose(inputs, [0, 2, 3, 1])
    dqn = Sequential()
    l1 = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(84, 84, 4))
    l1.set_input(inputs)
    dqn.add(l1)
    dqn.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
    dqn.add(Flatten())
    dqn.add(Dense(256, activation='relu'))
    dqn.add(Dense(num_actions))
    return dqn


################################################################################
# Convolutional CIFAR-10 models
################################################################################

def build_mentor_model_conv_cifar10(load=False):
    model = Sequential()
    l1 = Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(32, 32, 3), activation='relu')
    l1.set_input(img_cifar)
    model.add(l1)
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    if load:
        print ("Loading saved model")
        model.load_weights('cifar10.h5')
    return model


def build_mentee_model_conv_cifar10():
    model = Sequential()
    l1 = Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(32, 32, 3))
    l1.set_input(img_cifar)
    model.add(l1)
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model
