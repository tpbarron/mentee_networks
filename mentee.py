from __future__ import print_function
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
import models

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_alpha(t):
    """
    Compute alpha based on itr t
    """
    pass


def compute_beta(t):
    """
    Compute alpha based on itr t
    """
    pass


def compute_gamma(t):
    """
    Compute alpha based on itr t
    """
    pass



sess = tf.Session()
K.set_session(sess)

img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

mentee_model = models.build_mentee_model_sequential()
mentee_preds = mentee_model(img)
# mentee_l1, mentee_preds = models.build_mentee_model(img)

# tensorflow optimizer and gradients wrt loss
# NOTE: order is important here. If these lines are moved below
# the instantiation of the mentor net, then this also computes
# gradients for the mentor! We don't want that.
opt = tf.train.AdamOptimizer()
loss = tf.reduce_mean(categorical_crossentropy(labels, mentee_preds))
labels_grads_and_vars = opt.compute_gradients(loss)

print ("Label grads and vars ops: ", len(labels_grads_and_vars))

# build mentor model
mentor_model = models.build_mentor_model_sequential(load=True)
mentor_preds = mentor_model(img)
# mentor_l1, mentor_preds = models.build_mentor_model(img, load=True)

# define a loss function between the hidden layer activations of each network
# TODO: modify this so that it can handle different sizes of layers
probe_l1_loss = tf.sqrt(
                    tf.reduce_mean(
                        tf.square(
                            tf.sub(
                                mentor_model.layers[0].W, #mentor_l1
                                mentee_model.layers[0].W  #mentee_l1
                            )
                        )
                    )
                )
probe_l1_grads = tf.gradients(probe_l1_loss, [mentee_model.layers[0].W])[0]

probe_out_loss = tf.sqrt(
                    tf.reduce_mean(
                        tf.square(
                            tf.sub(
                                mentor_model.layers[1].W, #mentor_l1
                                mentee_model.layers[1].W  #mentee_l1
                            )
                        )
                    )
                )
probe_out_grads = tf.gradients(probe_out_loss, [mentee_model.layers[1].W])[0]
print (probe_out_grads)

sess.run(tf.initialize_all_variables())

num_iterations = 10
with sess.as_default():
    for i in range(num_iterations):
        if i % 100 == 0:
            print ("Step: ", i)
        batch = mnist.train.next_batch(50)
        # now compute gradients of mentee
        # compute activations of mentor
        # apply update rule to grads
        # apply new grads

        gradients = [sess.run(g, feed_dict={img: batch[0], labels: batch[1]}) for g,v in labels_grads_and_vars]

        probe_l1_gradients = sess.run(probe_l1_grads, feed_dict={img: batch[0], labels: batch[1]})
        probe_out_gradients = sess.run(probe_l1_grads, feed_dict={img: batch[0], labels: batch[1]})

        # print (probe_l1_gradients)
        # print (probe_out_gradients)
        # do weighted update of grads and call apply gradients

        # train_step.run(feed_dict={img: batch[0],
        #                           labels: batch[1]})

    # print acc_value.eval(feed_dict={img: mnist.test.images,
    #                                 labels: mnist.test.labels})

# mentee_model.save("mentee.h5")


# opt = tf.train.AdamOptimizer()
# # Compute the gradients for a list of variables.
# grads_and_vars = opt.compute_gradients(loss)
#
# # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
# # need to the 'gradient' part, for example cap them, etc.
# #capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]
#
# # Ask the optimizer to apply the capped gradients.
# opt.apply_gradients(capped_grads_and_vars)
