import torch
import numpy as np
import math
import os
import pickle

class SetDataset():
    def __init__(
            self, 
            batchsize=16,
            max_size=100000
        ):
        self.true_data = torch.from_numpy(np.array([])).float()
        self.false_data = torch.from_numpy(np.array([])).float()

        self.true_length = 0
        self.false_length = 0
        self.batchsize = batchsize
        self.data_batchsize = batchsize//2
        self.counter = 0
        self.num_batches = 0
        self.list_max_size = max_size//2

    @staticmethod
    def _getfilenames(path):
        true_filename = os.path.join(path, 'true_data.pkl')
        false_filename = os.path.join(path, 'false_data.pkl')

        return true_filename, false_filename

    def save(self, path):
        true_filename, false_filename = self._getfilenames(path)
        if not os.path.exists(path):
            os.makedirs(path)

        with open(true_filename, "wb") as f:
            pickle.dump(self.true_data, f)

        with open(false_filename, "wb") as f:
            pickle.dump(self.false_data, f)

    def load(self, path):
        true_filename, false_filename = self._getfilenames(path)
        if not os.path.exists(true_filename):
            print('[SetDataset] No true data found. Nothing was loaded')
            return
        if not os.path.exists(false_filename):
            print('[SetDataset] No false data found. Nothing was loaded')
            return
        
        with open(true_filename, "rb") as f:
            self.true_data = pickle.load(f)

        with open(false_filename, "rb") as f:
            self.false_data = pickle.load(f)

    def _set_batch_num(self):
        # get number of batches to run through so we see all the true data at least once
        # randomly get negative samples
        self.num_batches = math.ceil(
            min(self.true_length, self.false_length)/(self.data_batchsize)
        )

    def add_true_files(self, file_list):
        # load data from a file for true data
        for file in file_list:
            data = np.load(file)
            data = torch.from_numpy(data).float()
            self.true_data = self.concatenate(self.true_data, data)
        self.true_length = len(self.true_data)
        self._set_batch_num()
        self.shuffle()
        self.counter = 0
    
    def add_false_files(self, file_list):
        # load data from a file for false data
        for file in file_list:
            data = np.load(file)
            data = torch.from_numpy(data).float()
            self.false_data = self.concatenate(self.false_data, data)
        self.false_length = len(self.false_data)
        self._set_batch_num()
        self.shuffle()
        self.counter = 0

    def add_true_data(self, data_list):
        data = torch.squeeze(
            torch.stack(data_list), 1
        )
        self.true_data = self.concatenate(data, self.true_data)
        if len(self.true_data) > self.list_max_size:
            self.true_data = self.true_data[:self.list_max_size]
        self.true_length = len(self.true_data)
        self._set_batch_num()
        self.counter = 0

    def add_false_data(self, data_list):
        data = torch.squeeze(
            torch.stack(data_list), 1
        )
        self.false_data = self.concatenate(data, self.false_data)
        if len(self.false_data) > self.list_max_size:
            self.false_data = self.false_data[:self.list_max_size]
        self.false_length = len(self.false_data)
        self._set_batch_num()
        self.counter = 0

    def shuffle(self):
        self.true_data = self.true_data[
            torch.randperm(self.true_length)
        ]
        self.false_data = self.false_data[
            torch.randperm(self.false_length)
        ]

    @staticmethod
    def _get_minibatch(index, data, minibatch_size):
        minibatch = np.array([])
        if (index + minibatch_size) > len(data):
            num_remaining = len(data) - index
            minibatch = data[index:]
            minibatch = SetDataset.concatenate(minibatch, data[:minibatch_size-num_remaining])
        else:
            minibatch = data[index:index+minibatch_size]

        return minibatch

    def get_batch(self, shuffle=False):

        if self.true_length == 0 or self.false_length == 0:
            return self._unibatch()

        false_batch = self._get_minibatch(
            self.false_index(),
            self.false_data,
            self.data_batchsize)
        true_batch = self._get_minibatch(
            self.true_index(),
            self.true_data,
            self.data_batchsize)
        labels = [0]*self.data_batchsize + [1]*self.data_batchsize
        labels = torch.from_numpy(np.array(labels))

        self.counter += 1

        data = self.concatenate(false_batch, true_batch)

        if shuffle:
            shuffle_idxs = torch.randperm(len(data))
            data = data[shuffle_idxs]
            labels = labels[shuffle_idxs]

        return data, labels

    def true_index(self):
        return (self.counter*self.data_batchsize) % self.true_length

    def false_index(self):
        return (self.counter*self.data_batchsize) % self.false_length

    def _unibatch(self):
        if self.true_length == 0:
            return self._get_minibatch(
                self.false_index(), 
                self.false_data, 
                self.batchsize
                ), torch.from_numpy(np.array([0]*self.batchsize))
        else:
            return self._get_minibatch(
                self.true_index(), 
                self.true_data, 
                self.batchsize
                ), torch.from_numpy(np.array([1]*self.batchsize))

    @staticmethod
    def concatenate(arr1, arr2):

        if len(arr1) == 0:
            return arr2
        else:
            return torch.cat((arr1, arr2), axis=0)
    