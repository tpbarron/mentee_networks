from __future__ import print_function
import tensorflow as tf
import keras
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
import models
import learning_rates
import data_abstraction
import mnist_data
import cifar_data
import probe_ops
import os
import sys

# initialize the tensorflow session
sess = tf.Session()
K.set_session(sess)

if (len(sys.argv) < 3):
    p = None
    mode = 'obedient'
else:
    p = int(sys.argv[1])
    mode = sys.argv[2]

# Some parameters
num_epochs= 30
subsample = p # the number of samples from each class to use
batch_size = min(10*subsample, 100) # please set this to less than or equal to 10*subsample
mentee_mode = mode
temperature = 0.9
# list of probes between the mentor and mentee by layer; 0-indexed
# the output probe does not need to be specified
probes = [
    (0, 0),
    (1, 1)
]

USE_CONV = True
MNIST = True # True for MNIST, False for CIFAR-10

if (MNIST):
    dataset = mnist_data.read_data_sets('MNIST_data', one_hot=True, reshape=(not USE_CONV))
    mentee_model = models.build_mentee_model_conv() if USE_CONV else models.build_mentee_model()
    img_input = models.img_conv if USE_CONV else models.img_dense
else:
    dataset = cifar_data.read_data_sets('CIFAR_data')
    mentee_model = models.build_mentee_model_conv_cifar10()
    img_input = models.img_cifar

mentee_preds = mentee_model.output

run_name = mentee_mode + "_mentee" + ("_conv" if USE_CONV else "") + ("_mnist" if MNIST else "_cifar10") + ("_" + str(subsample) if subsample is not None else "")
summary_name = run_name + "_accuracy"
model_save_name = run_name + ".h5"
data = data_abstraction.DataAbstration(dataset, batch_size, subsample)

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

# build mentor model
if MNIST:
    mentor_model = models.build_mentor_model_conv(load=True) if USE_CONV else models.build_mentor_model(load=True)
else:
    mentor_model = models.build_mentor_model_conv_cifar10(load=True)
mentor_preds = mentor_model.output

probe_gradients = probe_ops.get_gradient_ops(probes, mentee_model, mentor_model, img_input, batch_size, temperature)

# ops to compute the accuracy of the mentor and mentee
acc_value_mentor = categorical_accuracy(models.labels, mentor_preds)
acc_value_mentee = categorical_accuracy(models.labels, mentee_preds)

# create a summary for our mentee accuracy
count = len([d for d in os.listdir('logs/') if os.path.isdir(os.path.join('logs/', d))])+1
log_dir = os.path.join('logs/', str(count))
os.mkdir(log_dir)
tensorboard_writer = tf.train.SummaryWriter(log_dir, graph=tf.get_default_graph())
tf.scalar_summary(summary_name, acc_value_mentee)
summary_op = tf.merge_all_summaries()

sess.run(tf.initialize_all_variables())

def train_mentee(mentee_mode):
    output = []
    last_epoch = -1
    best_accuracy = 0.0
    i = 0
    while data.epochs < num_epochs:
        if data.epochs > last_epoch:
            acc = sess.run(acc_value_mentee, feed_dict={img_input: dataset.test.images, models.labels: dataset.test.labels})
            output.append("epoch: "+ str(data.epochs) + ", accuracy: " + str(acc))

            # perform tensorboard ops the operations, and write log
            summary = sess.run(summary_op, feed_dict={img_input: dataset.test.images, models.labels: dataset.test.labels})
            tensorboard_writer.add_summary(summary, i)

            if acc > best_accuracy:
                best_accuracy = acc
                mentee_model.save(model_save_name)

            last_epoch = data.epochs
            print ("Step: ", last_epoch, acc)

        batch = data.next_batch()

        # # perform tensorboard ops the operations, and write log
        # summary = sess.run(summary_op, feed_dict={img_input: dataset.test.images, models.labels: dataset.test.labels})
        # tensorboard_writer.add_summary(summary, i)

        # Compute all needed gradients
        gradients = [sess.run(g, feed_dict={img_input: batch[0], models.labels: batch[1]}) for g, v in labels_grads_and_vars]

        # compute all probes (w/o the softmax probe)
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

        n = learning_rates.compute_n(data.epochs)
        # scale by 1.0/n because these param are a*lr and lr will be applied in the gradient update
        a = learning_rates.compute_eta_alpha(data.epochs, mentee_mode)*1.0/n
        b = learning_rates.compute_eta_beta(data.epochs, mentee_mode)*1.0/n
        g = learning_rates.compute_eta_gamma(data.epochs, mentee_mode)*1.0/n

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

        i += 1

    acc = sess.run(acc_value_mentee, feed_dict={img_input: dataset.test.images,
                                    models.labels: dataset.test.labels})
    output.append("epoch: " + str(data.epochs+1) + ", accuracy: " + str(acc))
    if acc > best_accuracy:
        mentee_model.save(model_save_name)

    return output

print ("Running MNIST-"+str(subsample)+" with mode: " + mentee_mode)
train_mentee(mentee_mode)
