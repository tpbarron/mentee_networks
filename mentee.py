import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
#from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy

import models

sess = tf.Session()
K.set_session(sess)

img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

mentor_model = models.build_mentor_model(load=True)
mentor_preds = mentor_model(img)

mentee_model = models.build_mentee_model()
mentee_preds = mentee_model(img)

# TODO: do mentee training here,
# 1) implement temperature softmax
# 2) Implement probes
# 3) Try using simply using returned gradients from tf and adding necessary
#       influence

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
