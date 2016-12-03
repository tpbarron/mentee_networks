import random
import numpy as np

class DataAbstration(object):

    def __init__(self, data_obj, batch, n=None):
        self.data_obj = data_obj
        self.batch = batch
        self.n = n
        if not n is None:
            # store n of each class
            self.samples = self.sample_from_data()
            self.batch_ind = 0
        else:
            self.samples = None
        self.epochs = 0


    def sample_from_data(self):
        images = self.data_obj.train.images
        images_shape = images.shape
        labels = self.data_obj.train.labels
        labels_shape = labels.shape

        num_samples = images_shape[0]

        counts = [0]*labels.shape[1] # num output classes
        start = random.randint(0, num_samples-1)

        ins = []
        outs = []

        while sum(counts) < self.n*len(counts):
            # add pattern i to samples
            i = images[start%num_samples]
            o = labels[start%num_samples]

            o_ind = int(np.nonzero(o)[0])
            if (counts[o_ind] < self.n):
                counts[int(np.nonzero(o)[0])] += 1
                ins.append(i)
                outs.append(o)

            start+=1

        return (np.array(ins), np.array(outs))


    def next_batch(self):
        if (self.samples is None):
            return self.data_obj.train.next_batch(self.batch)

        if (self.batch > self.samples[0].shape[0]):
            # return the entire sample set
            self.epochs+=1
            return self.samples

        # slice samples
        start = self.batch_ind
        end = (self.batch_ind + self.batch)
        if end > self.samples[0].shape[0]:
            end = self.samples[0].shape[0]

        sliced_samples = (self.samples[0][start:end], self.samples[1][start:end])
        self.batch_ind = end if (end != self.samples[0].shape[0]) else 0
        if (self.batch_ind == 0):
            self.epochs+=1

        return sliced_samples
