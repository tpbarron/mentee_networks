from __future__ import print_function
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
import models
import learning_rates
import os

# initialize the tensorflow session
sess = tf.Session()
K.set_session(sess)

USE_CONV = True

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=(not USE_CONV))

img_input = models.img_conv if USE_CONV else models.img_dense

# Some parameters
num_epochs= 100
batch_size = 100
dataset_config = ''
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

def train_mentee(dataset_config, mentee_mode):
    output =[]

    # #TODO: please check if I initialize the weights for the mentee network the following line correctly -- not affecting the mentor network
    # sess.run(tf.initialize_all_variables())

    mnist.train._index_in_epoch = 0 #re-read from the begining of the dataset

    # I think these dataset configs need to subsample the dataset, not
    # if dataset_config == 'mnist-1':
    #     total_batch = int(10 / batch_size)
    # elif dataset_config == 'mnist-10':
    #     total_batch = int(100 / batch_size)
    # elif dataset_config == 'mnist-50':
    #     total_batch = int(500 / batch_size)
    # elif dataset_config == 'mnist-100':
    #     total_batch = int(1000 / batch_size)
    # elif dataset_config == 'mnist-250':
    #     total_batch = int(2500 / batch_size)
    # elif dataset_config == 'mnist-500':
    #     total_batch = int(5000 / batch_size)
    # else:
    #     print ("the dataset configuration is undefined")
    #     import sys
    #     sys.exit()

    print ("dataset configuration: ", dataset_config)
    # print ("total number of batches: ", total_batch)
    print ("Mentee network mode ", mentee_mode)

    while mnist.train.epochs_completed < num_epochs:
        #mnist.train._index_in_epoch = 0 #re-read from the begining of the dataset
        acc = sess.run(acc_value_mentee, feed_dict={img_input: mnist.test.images, models.labels: mnist.test.labels})

        output.append("epoch: "+ str(mnist.train.epochs_completed) + ", accuracy: " + str(acc))
        print("Mentee accuracy at epoch: ", mnist.train.epochs_completed, acc,
              "epochs: ", mnist.train.epochs_completed, ", saving model")
        model_name = "mentee_conv.h5" if USE_CONV else "mentee_dense.h5"
        mentee_model.save(model_name)

        # #compute the learning rates.
        # alpha = 1000 * learning_rates.compute_eta_alpha(mnist.train.epochs_completed, mentee_mode)
        # beta = 1000 * learning_rates.compute_eta_beta(mnist.train.epochs_completed, mentee_mode)
        # gamma = 1000 * learning_rates.compute_eta_gamma(mnist.train.epochs_completed, mentee_mode)


        # for i in range(total_batch):
        #print("alpha: ", alpha, ", beta: ", beta, ", gamma: ", gamma)
        batch = mnist.train.next_batch(batch_size)
        #print ("batch ", batch[1])


        # Compute all needed gradients
        gradients = [sess.run(g, feed_dict={img_input: batch[0], models.labels: batch[1]}) for g,v in labels_grads_and_vars]

        # compute all probe (w/o the softmax probe)
        computed_probe_gradients = []
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

        n = learning_rates.compute_n(mnist.train.epochs_completed)
        a = learning_rates.compute_eta_alpha(mnist.train.epochs_completed, mentee_mode)
        b = learning_rates.compute_eta_beta(mnist.train.epochs_completed, mentee_mode)
        g = learning_rates.compute_eta_gamma(mnist.train.epochs_completed, mentee_mode)

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
        grads_n_vars = [(gradients[x], labels_grads_and_vars[x][1]) for x in range(len(labels_grads_and_vars))]
        sess.run(opt.apply_gradients(grads_n_vars), feed_dict={learning_rate: n})

    acc= sess.run(acc_value_mentee, feed_dict={img_input: mnist.test.images,
                                    models.labels: mnist.test.labels})
    output.append("epoch: " + str(mnist.train.epochs_completed+1) + ", accuracy: " + str(acc))
    #print (acc)

    model_name = "mentee_conv.h5" if USE_CONV else "mentee_dense.h5"
    mentee_model.save(model_name)
    return output


if __name__ == "__main__":

    train_mentee('', 'obedient')

    # for i in ("adamant", "obedient", "independent"):
    #     output = []
    #     file_name= (i + "_mentee_mode.txt")
    #     print (file_name)
    #     f = open(file_name, 'w')
    #     f.write(("Parameters: \n"
    #              + "num_epochs: 10" + ", batch_size: 50\n" ))
    #
    #
    #
    #     for j in ("mnist-1", "mnist-10", "mnist-50", "mnist-100", "mnist-250", "mnist-500"):
    #
    #         #mentee_model = models.build_mentee_model_conv() if USE_CONV else models.build_mentee_model()
    #         output= train_mentee(dataset_config=j, mentee_mode=i)
    #         f.write(("\ndataset configuration: " + j + "\n"))
    #         for line in output:
    #             f.write(str(line))
    #             f.write("\n")
    #
    #         #print ("The mentee accuracy for mentee mode: ", i, "and dataset configuration: ", j ," is : %", 100 * acc)
    #     f.close()
