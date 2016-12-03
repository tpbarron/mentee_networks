from __future__ import print_function

import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy

import models
import learning_rates
import mnist_data
import cifar_data

sess = tf.Session()
K.set_session(sess)

USE_CONV = True
MNIST = True # True for MNIST, False for CIFAR-10

if (MNIST):
    dataset = mnist_data.read_data_sets('MNIST_data', one_hot=True, reshape=(not USE_CONV))
    mentor_model = models.build_mentor_model_conv() if USE_CONV else models.build_mentor_model()
    img_input = models.img_conv if USE_CONV else models.img_dense
else:
    dataset = cifar_data.read_data_sets('CIFAR_data')
    mentor_model = models.build_mentor_model_conv_cifar10()
    img_input = models.img_cifar

mentor_preds = mentor_model.output
loss = tf.reduce_mean(categorical_crossentropy(models.labels, mentor_preds))

acc_value = categorical_accuracy(models.labels, mentor_preds)
learning_rate = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
sess.run(tf.initialize_all_variables())

num_epochs = 50
batch_size = 100

with sess.as_default():
    last_epoch = -1
    while dataset.train.epochs_completed < num_epochs:
        if dataset.train.epochs_completed > last_epoch:
            acc = acc_value.eval(feed_dict={img_input: dataset.test.images,
                                            models.labels: dataset.test.labels})
            last_epoch = dataset.train.epochs_completed
            print ("Step: ", last_epoch, acc)

        batch = dataset.train.next_batch(batch_size)
        n = learning_rates.compute_n(dataset.train.epochs_completed)
        train_step.run(feed_dict={img_input: batch[0],
                                  models.labels: batch[1],
                                  learning_rate: n})

    print (acc_value.eval(feed_dict={img_input: dataset.test.images,
                                    models.labels: dataset.test.labels}))

model_name = "mentor_conv.h5" if USE_CONV else "mentor_dense.h5"
mentor_model.save(model_name)
