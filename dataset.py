# -*- coding: utf-8 -*-
import numpy as np


def extract_images(data):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    data = data.reshape(data.shape[0], -1)
    return data


class DataSet(object):
    def __init__(self, images, labels, fake_data=0):
        assert images.shape[0] == labels.shape[0]
        self._num_examples = images.shape[0]
        images = images.astype(np.float32)
        self._images = np.where(np.isnan(images), 0, images)
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

        if fake_data:
            #self.images = [numpy.random.rand() for i in enumerate(self.images)]
            #numpy.where(1, self.images, numpy.random.rand())
            for i in range(self.images.shape[0]):
               # self.labels[i] = numpy.sum(self.images[i,:])
               self.labels[i] = np.sum(self.images[i])/100



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


def read_data_sets(images, labels, test_images, test_labels, train_ratio=0.75, fake_data=0):
    class DataSets(object):
        pass

    data_sets = DataSets()
    TRAIN_SIZE = int(images.shape[0] * train_ratio)
    Train_Images = extract_images(images)
    Train_Labels = labels

    test_images = extract_images(test_images)
    test_labels = test_labels

    perm = np.arange(images.shape[0])
    np.random.shuffle(perm)
    Train_Images = Train_Images[perm]
    Train_Labels = Train_Labels[perm]

    train_images = Train_Images[:TRAIN_SIZE]
    train_labels = Train_Labels[:TRAIN_SIZE]
    validation_images = Train_Images[TRAIN_SIZE:]
    validation_labels = Train_Labels[TRAIN_SIZE:]

    data_sets.train = DataSet(train_images, train_labels, fake_data)
    data_sets.validation = DataSet(validation_images, validation_labels, fake_data)
    data_sets.test = DataSet(test_images, test_labels, fake_data)

    print("train_size = {}  test_size = {}".format(images.shape[0], test_images.shape[0]))
    return data_sets

