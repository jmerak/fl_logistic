import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.examples.tutorials.mnist import input_data


class BatchGenerator:
    def __init__(self, x, yy):
        self.x = x
        self.y = yy
        self.size = len(x)
        self.random_order = list(range(len(x)))
        np.random.shuffle(self.random_order)
        self.start = 0
        return

    def next_batch(self, batch_size):
        perm = self.random_order[self.start:self.start + batch_size]

        self.start += batch_size
        if self.start > self.size:
            self.start = 0

        return self.x[perm], self.y[perm]

    # support slice
    def __getitem__(self, val):
        return self.x[val], self.y[val]


class Dataset(object):
    def __init__(self, load_data_func, one_hot=True, split = 0):
        minist = input_data.read_data_sets('data/', one_hot=True)
        x_train = minist.train.images
        y_train = minist.train.labels
        x_test = minist.test.images
        y_test = minist.test.labels
        print("Dataset: train-%d, test-%d" % (len(x_train), len(x_test)))

        if split == 0:
            self.train = BatchGenerator(x_train, y_train)
        else:
            self.train = self.splited_batch(x_train, y_train, split)

        self.test = BatchGenerator(x_test, y_test)

    def splited_batch(self, x_data, y_data, split):
        res = []
        for x, y in zip(np.split(x_data, split), np.split(y_data, split)):
            assert len(x) == len(y)
            res.append(BatchGenerator(x, y))
        return res
