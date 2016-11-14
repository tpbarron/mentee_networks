from __future__ import print_function
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
import models

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_alpha(epoch):
    """
    Compute alpha based on itr t
    """
    return 0.015


def compute_beta(epoch):
    """
    Compute alpha based on itr t
    """
    return 0.04


def compute_gamma(epoch):
    """
    Compute alpha based on itr t
    """
    return 0.01



sess = tf.Session()
K.set_session(sess)

mentee_model = models.build_mentee_model_sequential()
mentee_preds = mentee_model(models.img)

# tensorflow optimizer and gradients wrt loss
# NOTE: order is important here. If these lines are moved below
# the instantiation of the mentor net, then this also computes
# gradients for the mentor! We don't want that.
opt = tf.train.AdamOptimizer()
loss = tf.reduce_mean(categorical_crossentropy(models.labels, mentee_preds))
labels_grads_and_vars = opt.compute_gradients(loss)
apply_grads_and_vars = opt.apply_gradients(labels_grads_and_vars)

print ("Label grads and vars ops: ", len(labels_grads_and_vars))

# build mentor model
mentor_model = models.build_mentor_model_sequential(load=False)
mentor_preds = mentor_model(models.img)

# define a loss function between the hidden layer activations of each network
# TODO: modify this so that it can handle different sizes of layers
# TODO: loss needs to be defined between activations NOT weights
# TODO: this works because there are only two layers, not sure how to get activation of ith hidden layer not first or last
#       Try the K.function([...])
#       I think it will work to use:
#           mentor_preds_li = K.function([models.img], [mentor_model.layers[i].output]).outputs[0]
#           mentee_preds_li = K.function([models.img], [mentee_model.layers[i].output]).outputs[0]

mentor_l1_preds = mentor_model.layers[0](models.img)
mentee_l1_preds = mentee_model.layers[0](models.img)
probe_l1_loss = tf.sqrt(
                    tf.reduce_mean(
                        tf.square(
                            tf.sub(
                                mentor_l1_preds,
                                mentee_l1_preds
                            )
                        )
                    )
                )

probe_l1_grads = tf.gradients(probe_l1_loss, [mentee_model.layers[0].W, mentee_model.layers[0].b])
print (probe_l1_grads)

probe_out_loss = tf.sqrt(
                    tf.reduce_mean(
                        tf.square(
                            tf.sub(
                                mentor_preds,
                                mentee_preds,
                            )
                        )
                    )
                )
probe_out_grads = tf.gradients(probe_out_loss, [mentee_model.layers[0].W, mentee_model.layers[0].b, mentee_model.layers[1].W, mentee_model.layers[1].b])
print (probe_out_grads)

acc_value = categorical_accuracy(models.labels, mentee_preds)

sess.run(tf.initialize_all_variables())

num_iterations = 1000
with sess.as_default():
    for i in range(num_iterations):
        if i % 100 == 0:
            print ("Accuracy at step: ", i, acc_value.eval(feed_dict={models.img: mnist.test.images,
                                            models.labels: mnist.test.labels}),
                                            epochs: ", mnist,train.epochs_completed)

        batch = mnist.train.next_batch(100)
        # now compute gradients of mentee
        # compute activations of mentor
        # apply update rule to grads
        # apply new grads

        gradients = [sess.run(g, feed_dict={models.img: batch[0], models.labels: batch[1]}) for g,v in labels_grads_and_vars]
        probe_l1_gradients = [sess.run(g, feed_dict={models.img: batch[0], models.labels: batch[1]}) for g in probe_l1_grads]
        probe_out_gradients = [sess.run(g, feed_dict={models.img: batch[0], models.labels: batch[1]}) for g in probe_out_grads]

        # print (gradients[0])
        # print (len(probe_l1_gradients))
        # print (len(probe_out_gradients))

        a = compute_alpha(0)
        b = compute_beta(0)
        g = compute_gamma(0)

        # do weighted update of grads and call apply gradients
        # TODO: generalize this
        # gradients[0] is weights at hidden layers 0
        # gradients[1] is bias at hidden layer 0
        # gradients[2] is weights at hidden layer 1
        # gradients[3] is bias at hidden layer 1
        # gradients[i] is var at hidden layer i/2, this cooresponds to the probe_out_gradients[i]
        # but need probe_li_gradients[0,1] for each layer
        # probably want to generalize probe_l1_gradients to be probe_layer_gradients such that
        #    probe_layer_gradients[i] corresponds to the correct layer and probe_layer_gradients
        #    for the last two elements (aka the last layer) is 0
        # If this update can be made into a tensorflow op, then can just run train_step
        gradients[0] = a*gradients[0] + b*probe_l1_gradients[0] + g*probe_out_gradients[0]
        gradients[1] = a*gradients[1] + b*probe_l1_gradients[1] + g*probe_out_gradients[1]
        gradients[2] = a*gradients[2] + g*probe_out_gradients[2]
        gradients[3] = a*gradients[3] + g*probe_out_gradients[3]

        grads_n_vars = [(gradients[i], labels_grads_and_vars[i][1]) for i in range(len(labels_grads_and_vars))]
        sess.run(opt.apply_gradients(grads_n_vars))

        # train_step.run(feed_dict={models.img: batch[0],
        #                           models.labels: batch[1]})

    print (acc_value.eval(feed_dict={models.img: mnist.test.images,
                                    models.labels: mnist.test.labels}))

mentee_model.save("mentee.h5")


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
