from __future__ import print_function

import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
import os
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

run_name = "mentor" + ("_conv" if USE_CONV else "") + ("_mnist" if MNIST else "_cifar10")
summary_name = run_name + "_accuracy"
model_save_name = run_name + ".h5"

mentor_preds = mentor_model.output
loss = tf.reduce_mean(categorical_crossentropy(models.labels, mentor_preds))

acc_value = categorical_accuracy(models.labels, mentor_preds)
learning_rate = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# create a summary for our mentor accuracy
count = len([d for d in os.listdir('logs/') if os.path.isdir(os.path.join('logs/', d))])+1
log_dir = os.path.join('logs/', str(count))
os.mkdir(log_dir)
tensorboard_writer = tf.train.SummaryWriter(log_dir, graph=tf.get_default_graph())
tf.scalar_summary(summary_name, acc_value)
summary_op = tf.merge_all_summaries()

sess.run(tf.initialize_all_variables())

max_epochs = 30
batch_size = 100

with sess.as_default():
    last_epoch = -1
    best_accuracy = 0.0
    i = 0
    samples = dataset.train.images.shape[0]
    while dataset.train.epochs_completed < max_epochs:
        if dataset.train.epochs_completed > last_epoch:
            acc = acc_value.eval(feed_dict={img_input: dataset.test.images,
                                            models.labels: dataset.test.labels})
            if acc > best_accuracy:
                mentor_model.save(model_save_name)
                best_accuracy = acc

            last_epoch = dataset.train.epochs_completed
            # perform tensorboard ops the operations, and write log
            summary = sess.run(summary_op, feed_dict={img_input: dataset.test.images, models.labels: dataset.test.labels})
            tensorboard_writer.add_summary(summary, last_epoch) # i * batch_size is num samples seen #dataset.train.epochs_completed)

            print ("Epoch: " + str(last_epoch) + " of " + str(max_epochs) + ", accuracy: " + str(acc))

        # # # perform tensorboard ops the operations, and write log
        # summary = sess.run(summary_op, feed_dict={img_input: dataset.test.images, models.labels: dataset.test.labels})
        # print ("x = ", last_epoch + (i*batch_size)/float(samples))
        # tensorboard_writer.add_summary(summary, last_epoch+(i*batch_size)/float(samples)) # i * batch_size is num samples seen #dataset.train.epochs_completed)

        batch = dataset.train.next_batch(batch_size)
        n = learning_rates.compute_n(dataset.train.epochs_completed)
        train_step.run(feed_dict={img_input: batch[0],
                                  models.labels: batch[1],
                                  learning_rate: n})
        i += 1

    final_acc = acc_value.eval(feed_dict={img_input: dataset.test.images,
                                    models.labels: dataset.test.labels})
    if final_acc > best_accuracy:
        mentor_model.save(model_save_name)
    print ("Final accurach: " + str(final_acc))
