To run mentor:

python mentor.py

This will run with the full MNIST dataset on a model defined in Keras as follows:

mentor_model = Sequential()
mentor_model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode='same', activation='relu', input_shape=(28, 28, 1)))
mentor_model.add(Flatten())
mentor_model.add(Dense(10))
mentor_model.add(Activation('softmax'))

This achieves approximately 98% accuracy.

To run the mentee:

python mentee.py p mode

Where p is the number of samples for each class to train with and mode is
  "obedient", "adamant", or "independent"

The model for the mentee is almost the same but with only 16 filters in the convolutional layer

mentee_model = Sequential()
mentee_model.add(Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='same', activation='relu', input_shape=(28, 28, 1)))
mentee_model.add(Flatten())
mentee_model.add(Dense(10))
mentee_model.add(Activation('softmax'))
return mentee_model

Please see the plots/ directory to see graphs of the different runs. You can also view the runs with tensorboard if you'd like. Or if
you run them yourself the scripts will print out the log dir upon completion. You can then run `tensorboard --logdir=[logdir]`.

Currently all the runs will do 30 epochs.
