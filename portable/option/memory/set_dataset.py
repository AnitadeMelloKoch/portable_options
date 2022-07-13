import torch
import numpy as np
import math

class SetDataset():
    def __init__(self, batchsize=16):
        self.true_data = torch.from_numpy(np.array([])).float()
        self.false_data = torch.from_numpy(np.array([])).float()

        self.true_length = 0
        self.false_length = 0
        self.batchsize = batchsize
        self.data_batchsize = batchsize//2
        self.counter = 0
        self.num_batches = 0

    def add_true_files(self, file_list):
        for file in file_list:
            data = np.load(file)
            data = torch.from_numpy(data).float()
            self.true_data = self.concatenate(self.true_data, data)
        self.true_length = len(self.true_data)
        self.num_batches = math.ceil(
            min(self.true_length, self.false_length)/(self.data_batchsize)
        )
        self.shuffle()
        self.counter = 0
    
    def add_false_files(self, file_list):
        for file in file_list:
            data = np.load(file)
            data = torch.from_numpy(data).float()
            self.false_data = self.concatenate(self.false_data, data)
        self.false_length = len(self.false_data)
        self.num_batches = math.ceil(
            min(self.true_length, self.false_length)/(self.data_batchsize)
        )
        self.shuffle()
        self.counter = 0

    def shuffle(self):
        self.true_data = self.true_data[
            torch.randperm(self.true_length)
        ]
        self.false_data = self.false_data[
            torch.randperm(self.false_length)
        ]

    def batch_num(self):
        return self.num_batches

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
        false_index = (self.counter*self.data_batchsize) % self.false_length
        true_index = (self.counter*self.data_batchsize) % self.true_length
        false_batch = self._get_minibatch(false_index,self.false_data,self.data_batchsize)
        true_batch = self._get_minibatch(true_index,self.true_data,self.data_batchsize)
        labels = [0]*self.data_batchsize + [1]*self.data_batchsize
        labels = torch.from_numpy(np.array(labels))

        self.counter += 1

        data = self.concatenate(false_batch, true_batch)

        if shuffle:
            shuffle_idxs = torch.randperm(len(data))
            data = data[shuffle_idxs]
            labels = labels[shuffle_idxs]

        return data, labels

    @staticmethod
    def concatenate(arr1, arr2):

        if len(arr1) == 0:
            return arr2
        else:
            return torch.cat((arr1, arr2), axis=0)
    