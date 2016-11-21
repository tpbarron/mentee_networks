from __future__ import print_function
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
import models
import rates

# initialize the tensorflow session
sess = tf.Session()
K.set_session(sess)

USE_CONV = True

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=(not USE_CONV))

img_input = models.img_conv if USE_CONV else models.img_dense

# Some parameters
num_iterations = 5000
batch_size = 100
temperature = 0.9
# list of probes between the mentor and mentee by layer; 0-indexed
# the output probe does not need to be specified
probes = [
    (0, 0),
    (1, 1)
]

# a function that returns a list of gradient ops and perhaps the layer of the probe
def get_gradient_ops(probes, mentee, mentor, img_input, emperature):
    probe_gradients = []
    for p in probes:
        mentor_layer_index, mentee_layer_index = p

        # Ensure that the probe index is not the softmax layer
        assert mentor_layer_index < (len(mentor.layers) - 1)
        assert mentee_layer_index < (len(mentee.layers) - 1)

        # define a loss function between the hidden layer activations of each network
        # TODO: checkif this layer is convolutional, if so, need to use output shape
        # print (dir(mentor_model.layers[mentor_layer_index]))
        # print (mentor_model.layers[mentor_layer_index].output_shape)
        # import sys
        # sys.exit()
        mentor_layer_preds = K.function([img_input], [mentor_model.layers[mentor_layer_index].output]).outputs[0]
        mentee_layer_preds = K.function([img_input], [mentee_model.layers[mentee_layer_index].output]).outputs[0]

        if (isinstance(mentor_model.layers[mentor_layer_index], keras.layers.core.Dense)):
            # Dense case
            if (not isinstance(mentee_model.layers[mentee_layer_index], keras.layers.core.Dense)):
                raise Exception("Probes between two different layer types, Dense to Conv2d")

            output_mentor = mentor_model.layers[mentor_layer_index].output_dim
            output_mentee = mentee_model.layers[mentee_layer_index].output_dim
            if (output_mentee > output_mentor):
                raise Exception("Mentee has more outputs than mentor")
            slice_ind = output_mentee

            # Ok let's assume that the mentor will always be >= in size than the mentee
            # then want to consider the error along the first n value where n is the min(mentor, mentee)
            # use slice to ignore irrelevant outputs
            probe_layer_loss = tf.sqrt(
                                tf.reduce_mean(
                                    tf.squared_difference(
                                        tf.slice(mentor_layer_preds, [0, 0], [batch_size, slice_ind]), #mentee_model.layers[mentor_layer_index].output_dim]),
                                        tf.slice(mentee_layer_preds, [0, 0], [batch_size, slice_ind]) #mentee_model.layers[mentee_layer_index].output_dim])
                                    )
                                )
                            )
        elif (isinstance(mentor_model.layers[mentor_layer_index], keras.layers.convolutional.Convolution2D)):
            # Convolution2D case
            if (not isinstance(mentee_model.layers[mentee_layer_index], keras.layers.convolutional.Convolution2D)):
                raise Exception("Probes between two different layer types, Conv2d to Dense")

            out_shape_mentor = mentor_model.layers[mentor_layer_index].output_shape
            out_shape_mentee = mentee_model.layers[mentee_layer_index].output_shape
            if (out_shape_mentee[-1] > out_shape_mentor[-1]):
                raise Exception("Mentee has more feature maps than mentor")
            _, fmap_rows, fmap_cols, slice_ind = out_shape_mentee
            # Now assume that the feature maps in both nets are the same size and only the number
            # of feature maps vary
            probe_layer_loss = tf.sqrt(
                                tf.reduce_mean(
                                    tf.squared_difference(
                                        tf.slice(mentor_layer_preds, [0, 0, 0, 0], [batch_size, fmap_rows, fmap_cols, slice_ind]),
                                        tf.slice(mentee_layer_preds, [0, 0, 0, 0], [batch_size, fmap_rows, fmap_cols, slice_ind])
                                    )
                                )
                            )

        probe_layer_grads = tf.gradients(probe_layer_loss, mentee_model.trainable_weights)
        probe_gradients.append(probe_layer_grads)

    # added all probes, now add the output softmax probe
    # use the output of the last layer before the sotfmax
    # NOTE: assume this is a dense layer and that it has the same number of outputs
    mentor_out_preds = K.function([img_input], [mentor_model.layers[-2].output]).outputs[0]
    mentee_out_preds = K.function([img_input], [mentee_model.layers[-2].output]).outputs[0]

    # Define the loss between the mentor and mentee output with temperature softmax
    probe_out_loss = tf.sqrt(
                        tf.reduce_mean(
                            tf.squared_difference(
                                tf.div(tf.exp(tf.div(mentor_out_preds, temperature)), tf.reduce_sum(tf.exp(tf.div(mentor_out_preds, temperature)))),
                                tf.div(tf.exp(tf.div(mentee_out_preds, temperature)), tf.reduce_sum(tf.exp(tf.div(mentee_out_preds, temperature))))
                            )
                        )
                    )
    probe_out_grads = tf.gradients(probe_out_loss, mentee_model.trainable_weights)
    probe_gradients.append(probe_out_grads)
    return probe_gradients


# def compute_alpha(epoch):
#     """
#     Compute alpha based on itr t
#     """
#     return 0.1
#
#
# def compute_beta(epoch):
#     """
#     Compute alpha based on itr t
#     """
#     return 0.1
#
#
# def compute_gamma(epoch):
#     """
#     Compute alpha based on itr t
#     """
#     return 0.1


mentee_model = models.build_mentee_model_conv() if USE_CONV else models.build_mentee_model()
mentee_preds = mentee_model.output

# tensorflow optimizer and gradients wrt loss
# NOTE: order is important here. If these lines are moved below
# the instantiation of the mentor net, then this also computes
# gradients for the mentor! We don't want that.
learning_rate = tf.placeholder(tf.float32, shape=[])
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
loss = tf.reduce_mean(categorical_crossentropy(models.labels, mentee_preds))
labels_grads_and_vars = opt.compute_gradients(loss)
apply_grads_and_vars = opt.apply_gradients(labels_grads_and_vars)

print ("Label grads and vars ops: ", len(labels_grads_and_vars))
sess.run(tf.initialize_all_variables())

# build mentor model
mentor_model = models.build_mentor_model_conv(load=True) if USE_CONV else models.build_mentor_model(load=True)
mentor_preds = mentor_model.output

# ops to compute the accuracy of the mentor and mentee
acc_value_mentor = categorical_accuracy(models.labels, mentor_preds)
acc_value_mentee = categorical_accuracy(models.labels, mentee_preds)

probe_gradients = get_gradient_ops(probes, mentee_model, mentor_model, img_input, temperature)
print ("Probe gradients: ", probe_gradients)

for i in range(num_iterations):
    if i % 10 == 0:
        print ("Mentee accuracy at step: ", i, sess.run(acc_value_mentee, feed_dict={img_input: mnist.test.images,
                                        models.labels: mnist.test.labels}),
                                        "epochs: ", mnist.train.epochs_completed, ", saving model")
        model_name = "mentee_conv.h5" if USE_CONV else "mentee_dense.h5"
        mentee_model.save(model_name)

    batch = mnist.train.next_batch(batch_size)

    # Compute all needed gradients
    gradients = [sess.run(g, feed_dict={img_input: batch[0], models.labels: batch[1]}) for g,v in labels_grads_and_vars]

    # compute all probe (w/o the softmax probe)
    computed_probe_gradients = []
    for j in range(len(probe_gradients)-1):
        probe_grad = []
        probe_grad_op = probe_gradients[j]
        for g in probe_grad_op:
            if g is not None:
                probe_grad.append(sess.run(g, feed_dict={img_input: batch[0], models.labels: batch[1]}))
            else:
                probe_grad.append(None)
        computed_probe_gradients.append(probe_grad)

    # compute gradients for softmax probe
    computed_probe_out_gradients = [sess.run(g, feed_dict={img_input: batch[0], models.labels: batch[1]}) for g in probe_gradients[-1]]

    n = rates.compute_n(mnist.train.epochs_completed)
    a = rates.compute_alpha(mnist.train.epochs_completed)
    b = rates.compute_beta(mnist.train.epochs_completed)
    g = rates.compute_gamma(mnist.train.epochs_completed)

    for j in range(len(gradients)):
        # set gradients for variable j
        gradients[j] = a*gradients[j]

        # sum the probes
        for k in range(len(computed_probe_gradients)):
            probe_grad = computed_probe_gradients[k]
            if (probe_grad[j] is not None): # if there is a gradient from probe k for var j
                gradients[j] += b*probe_grad[j]

        # add the output softmax probe
        gradients[j] += g*computed_probe_out_gradients[j]

    # apply grads
    grads_n_vars = [(gradients[i], labels_grads_and_vars[i][1]) for i in range(len(labels_grads_and_vars))]
    sess.run(opt.apply_gradients(grads_n_vars), feed_dict={learning_rate: n})


print (sess.run(acc_value_mentee, feed_dict={img_input: mnist.test.images,
                                models.labels: mnist.test.labels}))

model_name = "mentee_conv.h5" if USE_CONV else "mentee_dense.h5"
mentee_model.save(model_name)
