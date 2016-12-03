# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Functions for downloading and reading CIFAR-10 data.
This is heavily modified from the TensorFlow CIFAR-10 example and includes some
Keras utilities as well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import cPickle
import sys
import os
import numpy as np
from keras.utils import np_utils


from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

SOURCE_URL = 'http://www.cs.toronto.edu/~kriz/'
FILE_NAME = 'cifar-10-python.tar.gz'

class DataSet(object):

    def __init__(self,
               images,
               labels,
               dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

        print ("Shapes: ", images.shape, labels.shape)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

        self._num_examples = images.shape[0]
        # print (np.mean(images))
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        # print (np.mean(images))

        self._images = images
        self._labels = labels
        print ("labels: ", labels.shape)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]
        # Start next epoch
        start = 0
        self._index_in_epoch = batch_size
        assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding="bytes")
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode("utf8")] = v
        d = d_decoded
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data(train_dir):
    dirname = "cifar-10-batches-py"
    path = os.path.join(train_dir, dirname)

    nb_train_samples = 50000

    X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        X_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    X_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # tensorflow dim ordering
    X_train = X_train.transpose(0, 2, 3, 1)
    X_test = X_test.transpose(0, 2, 3, 1)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return (X_train, y_train), (X_test, y_test)


def read_data_sets(train_dir,
                   dtype=dtypes.float32):

    print ("Maybe downloading " + str(SOURCE_URL+FILE_NAME) + "...")
    local_file = base.maybe_download(FILE_NAME, train_dir,
                                   SOURCE_URL+FILE_NAME)
    print ("Finished downloading.")
    if (local_file.endswith("tar.gz")):
        tar = tarfile.open(local_file, "r:gz")
        tar.extractall(path=train_dir)
        tar.close()

    train_data, test_data = load_data(train_dir)
    train = DataSet(train_data[0], train_data[1], dtype=dtype)
    test = DataSet(test_data[0], test_data[1], dtype=dtype)
    return base.Datasets(train=train, validation=None, test=test)


def load_cifar(train_dir='CIFAR_data'):
    return read_data_sets(train_dir)
