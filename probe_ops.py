import tensorflow as tf
from keras import backend as K
import keras


# a function that returns a list of gradient ops and perhaps the layer of the probe
def get_gradient_ops(probes, mentee_model, mentor_model, img_input, batch_size, temperature):
    probe_gradients = []
    for p in probes:
        mentor_layer_index, mentee_layer_index = p

        # Ensure that the probe index is not the softmax layer
        assert mentor_layer_index < (len(mentor_model.layers) - 1)
        assert mentee_layer_index < (len(mentee_model.layers) - 1)

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
