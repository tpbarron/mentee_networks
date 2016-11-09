import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy

import models

sess = tf.Session()
K.set_session(sess)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

mentor_model = models.build_mentor_model()
mentor_preds = mentor_model(img)

loss = tf.reduce_mean(categorical_crossentropy(labels, mentor_preds))

acc_value = categorical_accuracy(labels, mentor_preds)
train_step = tf.train.AdamOptimizer().minimize(loss)
sess.run(tf.initialize_all_variables())

with sess.as_default():
    for i in range(2000):
        if i % 100 == 0:
            print "Step: ", i
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})

    print acc_value.eval(feed_dict={img: mnist.test.images,
                                    labels: mnist.test.labels})

mentor_model.save("mentor.h5")
