import torch
import numpy as np
import math
import os
import pickle
from portable.utils import plot_state

class SetDataset():
    def __init__(
            self, 
            batchsize=16,
            max_size=100000
        ):
        self.true_data = torch.from_numpy(np.array([])).float()
        self.false_data = torch.from_numpy(np.array([])).float()
        self.priority_false_data = torch.from_numpy(np.array([])).float()

        self.true_length = 0
        self.false_length = 0
        self.priority_false_length = 0
        self.batchsize = batchsize
        self.data_batchsize = batchsize//2
        self.counter = 0
        self.num_batches = 0
        self.list_max_size = max_size//2

        self.shuffled_indices_true = None
        self.shuffled_indices_false = None
        self.shuffled_indices_false_priority = None

    def transform(self, x):
        return x/255.0

    @staticmethod
    def _getfilenames(path):
        true_filename = os.path.join(path, 'true_data.pkl')
        false_filename = os.path.join(path, 'false_data.pkl')
        priority_false_filename = os.path.join(path, 'priority_false_data.pkl')

        return true_filename, false_filename, priority_false_filename

    def save(self, path):
        true_filename, false_filename, priority_false_filename = self._getfilenames(path)
        if not os.path.exists(path):
            os.makedirs(path)

        with open(true_filename, "wb") as f:
            pickle.dump(self.true_data, f)

        with open(false_filename, "wb") as f:
            pickle.dump(self.false_data, f)

        with open(priority_false_filename, "wb") as f:
            pickle.dump(self.priority_false_data, f)

    def load(self, path):
        true_filename, false_filename, priority_false_filename = self._getfilenames(path)
        if not os.path.exists(true_filename):
            print('[SetDataset] No true data found. Nothing was loaded')
            return
        if not os.path.exists(false_filename):
            print('[SetDataset] No false data found. Nothing was loaded')
            return
        
        if not os.path.exists(false_filename):
            print('[SetDataset] No priority false data found. Nothing was loaded')
            return
        
        with open(true_filename, "rb") as f:
            self.true_data = pickle.load(f)
        
        self.true_length = len(self.true_data)

        with open(false_filename, "rb") as f:
            self.false_data = pickle.load(f)

        self.false_length = len(self.false_data)

        with open(priority_false_filename, "rb") as f:
            self.priority_false_data = pickle.load(f)

        self.priority_false_length = len(self.priority_false_data)

        self._set_batch_num()
        self.shuffle()

    def _set_batch_num(self):
        # get number of batches to run through so we see all the true data at least once
        # randomly get negative samples
        if self.true_length == 0 or self.false_length == 0:
            self.num_batches = math.ceil(
                max(self.true_length, self.false_length)/(self.data_batchsize)
            )
        else:
            self.num_batches = math.ceil(
                max(self.true_length, self.priority_false_length)/(self.data_batchsize)
            )

    def add_true_files(self, file_list):
        # load data from a file for true data
        for file in file_list:
            data = np.load(file)
            data = torch.from_numpy(data).float()
            data = data.squeeze()
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
            data = data.squeeze()
            self.false_data = self.concatenate(self.false_data, data)
        self.false_length = len(self.false_data)
        self._set_batch_num()
        self.shuffle()
        self.counter = 0

    def add_priority_false_files(self, file_list):
        # load data from a file for priority false data
        for file in file_list:
            data = np.load(file)
            data = torch.from_numpy(data).float()
            data = data.squeeze()
            self.priority_false_data = self.concatenate(self.priority_false_data, data)
        self.priority_false_length = len(self.priority_false_data)
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
        self.shuffle()

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
        self.shuffle()

    def add_priority_false_data(self, data_list):
        data = torch.squeeze(
            torch.stack(data_list), 1
        )
        self.priority_false_data = self.concatenate(data, self.priority_false_data)
        if len(self.priority_false_data) > self.list_max_size//2:
            self.priority_false_data = self.priority_false_data[:self.list_max_size//2]
        self.priority_false_length = len(self.priority_false_data)
        self._set_batch_num()
        self.counter = 0
        self.shuffle()

    def shuffle(self):
        self.shuffled_indices_true = torch.randperm(self.true_length)
        self.shuffled_indices_false = torch.randperm(self.false_length)
        self.shuffled_indices_false_priority = torch.randperm(self.priority_false_length)
        

    @staticmethod
    def _get_minibatch(index, data, minibatch_size, shuffled_indices):
        minibatch = np.array([])
        if (index + minibatch_size) > len(data):
            num_remaining = len(data) - index
            minibatch = data[shuffled_indices[index:]]
            minibatch = SetDataset.concatenate(minibatch, data[shuffled_indices[:minibatch_size-num_remaining]])
        else:
            minibatch = data[shuffled_indices[index:index+minibatch_size]]

        return minibatch

    def get_batch(self, shuffle_batch=True):

        if self.true_length == 0 or self.false_length == 0:
            return self._unibatch()

        if self.priority_false_length > 0:
            normal_false = self._get_minibatch(
                self.false_index(True),
                self.false_data,
                self.data_batchsize // 2,
                self.shuffled_indices_false
            )
            # print(torch.max(normal_false[0]))
            priority_false = self._get_minibatch(
                self.priority_false_index(),
                self.priority_false_data,
                self.data_batchsize - self.data_batchsize//2,
                self.shuffled_indices_false_priority
            )
            # print(torch.max(priority_false[0]))
            false_batch = self.concatenate(normal_false, priority_false)
        else:
            false_batch = self._get_minibatch(
                self.false_index(False),
                self.false_data,
                self.data_batchsize,
                self.shuffled_indices_false)
        true_batch = self._get_minibatch(
            self.true_index(),
            self.true_data,
            self.data_batchsize,
            self.shuffled_indices_true)
        # print(torch.max(true_batch[0]))

        labels = [0]*len(false_batch) + [1]*len(true_batch)
        labels = torch.from_numpy(np.array(labels))

        self.counter += 1

        data = self.concatenate(false_batch, true_batch)

        if shuffle_batch is True:
            shuffle_idxs = torch.randperm(len(data))
            data = data[shuffle_idxs]
            labels = labels[shuffle_idxs]

        # print("pre-transform data:", data)
        data = self.transform(data)
        # print("transformed data:", data)
        # print(labels)

        return data, labels

    def true_index(self):
        return (self.counter*self.data_batchsize) % self.true_length

    def false_index(self, use_priority_false):
        if use_priority_false:
            return (self.counter*self.data_batchsize//2) % self.false_length
        return (self.counter*self.data_batchsize) % self.false_length

    def priority_false_index(self):
        return (self.counter*(self.data_batchsize - self.data_batchsize//2)) % self.priority_false_length

    def _unibatch(self):
        if self.true_length == 0:
            data =  self._get_minibatch(
                self.false_index(False), 
                self.false_data, 
                self.batchsize
                ) 
            labels = torch.from_numpy(np.array([0]*len(data)))
        else:
            data = self._get_minibatch(
                self.true_index(), 
                self.true_data, 
                self.batchsize
                )
            labels = torch.from_numpy(np.array([1]*len(data)))
        
        return data, labels

    @staticmethod
    def concatenate(arr1, arr2):

        if len(arr1) == 0:
            return arr2
        else:
            return torch.cat((arr1, arr2), axis=0)
    