from __future__ import print_function
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
import models

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# initialize the tensorflow session
sess = tf.Session()
K.set_session(sess)


# Some parameters
num_iterations = 1000
batch_size = 100
temperature = 0.9
# list of probes between the mentor and mentee by layer; 0-indexed
# the output probe does not need to be specified
probes = [
    (0, 0),
    (1, 1)
]

# a function that returns a list of gradient ops and perhaps the layer of the probe
def get_gradient_ops(probes, mentee, mentor, temperature):
    probe_gradients = []
    for p in probes:
        mentor_layer_index, mentee_layer_index = p

        assert mentor_layer_index < (len(mentor.layers) - 1)
        assert mentee_layer_index < (len(mentee.layers) - 1)

        # define a loss function between the hidden layer activations of each network
        mentor_layer_preds = K.function([models.img], [mentor_model.layers[mentor_layer_index].output]).outputs[0]
        mentee_layer_preds = K.function([models.img], [mentee_model.layers[mentee_layer_index].output]).outputs[0]

        # Ok let's assume that the mentor will always be >= in size than the mentee
        # then want to consider the error along the first n value where n is the min(mentor, mentee)
        # use slice to ignore irrelevant outputs
        probe_layer_loss = tf.sqrt(
                            tf.reduce_mean(
                                tf.squared_difference(
                                    tf.slice(mentor_layer_preds, [0, 0], [batch_size, mentee_model.layers[mentor_layer_index].output_dim]),
                                    tf.slice(mentee_layer_preds, [0, 0], [batch_size, mentee_model.layers[mentee_layer_index].output_dim])
                                )
                            )
                        )
        probe_layer_grads = tf.gradients(probe_layer_loss, mentee_model.trainable_weights)
        probe_gradients.append(probe_layer_grads)

    # added all probes, now add the output softmax probe
    # use the output of the last layer before the sotfmax
    mentor_out_preds = K.function([models.img], [mentor_model.layers[-2].output]).outputs[0]
    mentee_out_preds = K.function([models.img], [mentee_model.layers[-2].output]).outputs[0]

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


def compute_alpha(epoch):
    """
    Compute alpha based on itr t
    """
    return 0.001


def compute_beta(epoch):
    """
    Compute alpha based on itr t
    """
    return 0.1


def compute_gamma(epoch):
    """
    Compute alpha based on itr t
    """
    return 0.1


mentee_model = models.build_mentee_model_sequential()
mentee_preds = mentee_model.output

# tensorflow optimizer and gradients wrt loss
# NOTE: order is important here. If these lines are moved below
# the instantiation of the mentor net, then this also computes
# gradients for the mentor! We don't want that.
opt = tf.train.AdamOptimizer()
loss = tf.reduce_mean(categorical_crossentropy(models.labels, mentee_preds))
labels_grads_and_vars = opt.compute_gradients(loss)
apply_grads_and_vars = opt.apply_gradients(labels_grads_and_vars)

print ("Label grads and vars ops: ", len(labels_grads_and_vars))
sess.run(tf.initialize_all_variables())

# build mentor model
mentor_model = models.build_mentor_model_sequential(load=True)
mentor_preds = mentor_model.output

# ops to compute the accuracy of the mentor and mentee
acc_value_mentor = categorical_accuracy(models.labels, mentor_preds)
acc_value_mentee = categorical_accuracy(models.labels, mentee_preds)

probe_gradients = get_gradient_ops(probes, mentee_model, mentor_model, temperature)
print ("Probe gradients: ", probe_gradients)

for i in range(num_iterations):
    if i % 100 == 0:
        print ("Mentee accuracy at step: ", i, sess.run(acc_value_mentee, feed_dict={models.img: mnist.test.images,
                                        models.labels: mnist.test.labels}),
                                        "epochs: ", mnist.train.epochs_completed, ", saving model")
        mentee_model.save("mentee.h5")

    batch = mnist.train.next_batch(batch_size)

    # Compute all needed gradients
    gradients = [sess.run(g, feed_dict={models.img: batch[0], models.labels: batch[1]}) for g,v in labels_grads_and_vars]

    # compute all probe (w/o the softmax probe)
    computed_probe_gradients = []
    for i in range(len(probe_gradients)-1):
        probe_grad = []
        probe_grad_op = probe_gradients[i]
        for g in probe_grad_op:
            if g is not None:
                probe_grad.append(sess.run(g, feed_dict={models.img: batch[0], models.labels: batch[1]}))
            else:
                probe_grad.append(None)
        computed_probe_gradients.append(probe_grad)

    # compute gradients for softmax probe
    computed_probe_out_gradients = [sess.run(g, feed_dict={models.img: batch[0], models.labels: batch[1]}) for g in probe_gradients[-1]]

    a = compute_alpha(i)
    b = compute_beta(i)
    g = compute_gamma(i)

    for i in range(len(gradients)):
        # set gradients for variable i
        gradients[i] = a*gradients[i]
        # sum the probes
        for j in range(len(computed_probe_gradients)):
            probe_grad = computed_probe_gradients[j]
            if (probe_grad[i] is not None): # if there is a gradient for probe j for layer i
                gradients[i] += b*probe_grad[i]

        # add the output softmax probe
        gradients[i] += computed_probe_out_gradients[i]

    # apply grads
    grads_n_vars = [(gradients[i], labels_grads_and_vars[i][1]) for i in range(len(labels_grads_and_vars))]
    sess.run(opt.apply_gradients(grads_n_vars))


print (sess.run(acc_value_mentee, feed_dict={models.img: mnist.test.images,
                                models.labels: mnist.test.labels}))

mentee_model.save("mentee.h5")
