from __future__ import print_function

import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy

import models
import learning_rates

sess = tf.Session()
K.set_session(sess)

USE_CONV = True

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=(not USE_CONV))

mentor_model = models.build_mentor_model_conv() if USE_CONV else models.build_mentor_model()
mentor_preds = mentor_model.output

loss = tf.reduce_mean(categorical_crossentropy(models.labels, mentor_preds))

acc_value = categorical_accuracy(models.labels, mentor_preds)
learning_rate = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
sess.run(tf.initialize_all_variables())

img_input = models.img_conv if USE_CONV else models.img_dense

num_epochs = 50
batch_size = 100

with sess.as_default():
    last_epoch = -1
    while mnist.train.epochs_completed < num_epochs:
        if mnist.train.epochs_completed > last_epoch:
            acc = acc_value.eval(feed_dict={img_input: mnist.test.images,
                                            models.labels: mnist.test.labels})
            last_epoch = mnist.train.epochs_completed
            print ("Step: ", last_epoch, acc)

        batch = mnist.train.next_batch(batch_size)
        n = learning_rates.compute_n(mnist.train.epochs_completed)
        train_step.run(feed_dict={img_input: batch[0],
                                  models.labels: batch[1],
                                  learning_rate: n})

    print (acc_value.eval(feed_dict={img_input: mnist.test.images,
                                    models.labels: mnist.test.labels}))

model_name = "mentor_conv.h5" if USE_CONV else "mentor_dense.h5"
mentor_model.save(model_name)
