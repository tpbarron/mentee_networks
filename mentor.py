from __future__ import print_function

import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy

import models

sess = tf.Session()
K.set_session(sess)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

mentor_model = models.build_mentor_model_sequential_conv()
mentor_preds = mentor_model.output

loss = tf.reduce_mean(categorical_crossentropy(models.labels, mentor_preds))

acc_value = categorical_accuracy(models.labels, mentor_preds)
train_step = tf.train.AdamOptimizer().minimize(loss)
sess.run(tf.initialize_all_variables())

with sess.as_default():
    for i in range(2000):
        if i % 100 == 0:
            acc = acc_value.eval(feed_dict={models.img: mnist.test.images,
                                            models.labels: mnist.test.labels})
            print ("Step: ", i, acc)
        batch = mnist.train.next_batch(10)
        train_step.run(feed_dict={models.img: batch[0],
                                  models.labels: batch[1]})

    print (acc_value.eval(feed_dict={models.img: mnist.test.images,
                                    models.labels: mnist.test.labels}))

mentor_model.save("mentor_conv.h5")
