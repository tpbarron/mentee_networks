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

  Where p is the number of samples for each class to train with [1, 10, 50, 100, 250, 500]
    (though other values should work, too) and mode is "obedient", "adamant", or "independent"

  The model for the mentee is almost the same but with only 16 filters in the convolutional layer

  mentee_model = Sequential()
  mentee_model.add(Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='same', activation='relu', input_shape=(28, 28, 1)))
  mentee_model.add(Flatten())
  mentee_model.add(Dense(10))
  mentee_model.add(Activation('softmax'))
  return mentee_model


If you'd like to run all the tests run:

  ./run.sh

  This will simply run all of mentees. It will NOT re run the mentor. By default the mentor model that is loaded is
  models/mentor_conv_mnist.h5

  This will generate 18 logs in logs/[mode]_logs/[mode]p.log where mode is the mentee mode and p is the MNIST-p


The first run should auto download MNIST data.

Any of the mentee runs will generate logs in /logs/[mode]_logs/[mode]p/. Once a log is there is needs to be deleted before the same one is run again.
A mentor run will generate an incremented log in /logs/[count]

Please see the plots/ directory to see graphs of the different runs. You can also view the runs with tensorboard if you'd like. Or if
you run them yourself the scripts will print out the log dir upon completion. You can then run `tensorboard --logdir=[logdir]`.

To view sample runs execute:
  `tensorboard --logdir=mentor:backup_logs/logs/43,adamant:backup_logs/logs/adamant_logs,obedient:backup_logs/logs/obedient_logs/,independent:backup_logs/logs/independent_logs`

Currently all the runs will do 30 epochs.

Finally, there is also a policy gradient implementation of mentee networks with the OpenAI gym environments in polgrad_mentee.py and polgrad_mentor.py
