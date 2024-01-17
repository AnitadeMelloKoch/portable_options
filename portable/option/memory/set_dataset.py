import torch
import numpy as np
import math
import os
import pickle
import random
from portable.utils import plot_state

class SetDataset():
    def __init__(
            self, 
            batchsize=16,
            max_size=100000,
            pad_func=lambda x: x,
            attention_num=8,
            create_validation_set=False,
            validation_set_size=0.3
        ):
        self.true_data = torch.from_numpy(np.array([])).float()
        self.false_data = torch.from_numpy(np.array([])).float()
        self.priority_false_data = torch.from_numpy(np.array([])).float()
        
        self.true_confidence = torch.from_numpy(np.array([])).float()
        self.false_confidence = torch.from_numpy(np.array([])).float()
        self.priority_false_confidence = torch.from_numpy(np.array([])).float()

        self.true_length = 0
        self.false_length = 0
        self.priority_false_length = 0

        self.true_train_length = 0
        self.false_train_length = 0
        self.priority_false_train_length = 0

        self.true_test_length = 0
        self.false_test_length = 0
        self.priority_false_test_length = 0
        
        self.batchsize = batchsize
        self.data_batchsize = batchsize//2
        self.pad = pad_func
        self.counter = 0
        self.num_batches = 0
        self.list_max_size = max_size//2

        self.attention_num = attention_num

        self.shuffled_indices_true = None
        self.shuffled_indices_false = None
        self.shuffled_indices_priority_false = None
        
        self.validate = create_validation_set
        self.val_size = validation_set_size
        self.validate_indicies_true = []
        self.validate_indicies_false = []
        self.validate_indicies_priority_false = []
        

    @staticmethod
    def transform(x):
        if torch.max(x) > 1:
            return x/255.0
        else:
            return x

    def set_transform_function(self, transform):
        self.transform = transform
    
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
            pickle.dump((self.true_data, self.true_confidence), f)

        with open(false_filename, "wb") as f:
            pickle.dump((self.false_data, self.false_confidence), f)

        with open(priority_false_filename, "wb") as f:
            pickle.dump((self.priority_false_data, self.priority_false_confidence), f)

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
            self.true_data, self.true_confidence = pickle.load(f)
        
        self.true_length = len(self.true_data)

        with open(false_filename, "rb") as f:
            self.false_data, self.false_confidence = pickle.load(f)

        self.false_length = len(self.false_data)

        with open(priority_false_filename, "rb") as f:
            self.priority_false_data, self.priority_false_confidence = pickle.load(f)

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
        # Have confidence weight 1 (full confidence)
        for file in file_list:
            data = np.load(file)
            data = self.pad(data)
            data = torch.from_numpy(data).float()
            data = data.squeeze()
            self.true_data = self.concatenate(self.true_data, data)
            self.true_confidence = self.concatenate(self.true_confidence, 
                                                    torch.ones(len(data), self.attention_num))
        self.true_length = len(self.true_data)
        self._set_batch_num()
        self.counter = 0
        
        if self.validate:
            self.validate_indicies_true = np.random.choice(range(0, self.true_length),
                                                      int(self.true_length*self.val_size),
                                                      replace=False)
        self.shuffle()

    def add_some_true_files(self, file_list):
        # load data from true file and only add a few random samples
        for file in file_list:
            data = np.load(file)
            data = self.pad(data)
            data = torch.from_numpy(data).float()
            data = data.squeeze()
            data = random.sample(data, 20)
            self.true_data = self.concatenate(self.true_data, data)
            self.true_confidence = self.concatenate(self.true_confidence, 
                                                    torch.ones(len(data), self.attention_num))
        self._set_batch_num()
        self.shuffle()
        self.counter = 0

    def add_false_files(self, file_list):
        # load data from a file for false data
        # From file, full confidence for false data
        for file in file_list:
            data = np.load(file)
            data = self.pad(data)
            data = torch.from_numpy(data).float()
            data = data.squeeze()
            self.false_data = self.concatenate(self.false_data, data)
            self.false_confidence = self.concatenate(self.false_confidence, 
                                                     torch.ones(len(data), self.attention_num))
        self.false_length = len(self.false_data)
        self._set_batch_num()
        self.counter = 0
        
        if self.validate:
            self.validate_indicies_false = np.random.choice(range(0, self.false_length),
                                                      int(self.false_length*self.val_size),
                                                      replace=False)
        self.shuffle()

    def add_some_false_files(self, file_list):
        # load data from true file and only add a few random samples
        for file in file_list:
            data = np.load(file)
            data = self.pad(data)
            data = torch.from_numpy(data).float()
            data = data.squeeze()
            data = random.sample(data, 20)
            self.false_data = self.concatenate(self.false_data, data)
            self.false_confidence = self.concatenate(self.false_confidence, 
                                                     torch.ones(len(data), self.attention_num))
        self.false_length = len(self.false_data)
        self._set_batch_num()
        self.shuffle()
        self.counter = 0
        
    
    def add_priority_false_files(self, file_list):
        # load data from a file for priority false data
        for file in file_list:
            data = np.load(file)
            data = self.pad(data)
            data = torch.from_numpy(data).float()
            data = data.squeeze()
            self.priority_false_data = self.concatenate(self.priority_false_data, data)
            self.priority_false_confidence = self.concatenate(self.priority_false_confidence, 
                                                              torch.ones(len(data), self.attention_num))
        self.priority_false_length = len(self.priority_false_data)
        self._set_batch_num()
        self.counter = 0

        if self.validate:
            self.validate_indicies_priority_false = np.random.choice(range(0, self.priority_false_length),
                                                    int(self.priority_false_length*self.val_size),
                                                    replace=False)
        self.shuffle()

    def add_true_data(self, data_list, conf_list):
        # assert confidence shape
        assert len(conf_list[0]) == self.attention_num

        data = torch.squeeze(
            torch.stack(data_list), 1
        )
        confidence = torch.from_numpy(np.array(conf_list)).float()
        self.true_data = self.concatenate(data, self.true_data)
        self.true_confidence = self.concatenate(confidence, self.true_confidence) 
        if len(self.true_data) > self.list_max_size:
            self.true_data = self.true_data[:self.list_max_size]
            self.true_confidence = self.true_confidence[:self.list_max_size]
        self.true_length = len(self.true_data)
        self._set_batch_num()
        self.counter = 0
        self.shuffle()

    def add_false_data(self, data_list, conf_list):
        # assert confidence shape
        assert len(conf_list[0]) == self.attention_num
        
        data = torch.squeeze(
            torch.stack(data_list), 1
        )
        confidence = torch.from_numpy(np.array(conf_list)).float()
        self.false_data = self.concatenate(data, self.false_data)
        self.false_confidence = self.concatenate(confidence, self.false_confidence)
        if len(self.false_data) > self.list_max_size:
            self.false_data = self.false_data[:self.list_max_size]
            self.priority_false_confidence = self.priority_false_confidence[:self.list_max_size]
        self.false_length = len(self.false_data)
        self._set_batch_num()
        self.counter = 0
        self.shuffle()

    def add_priority_false_data(self, data_list, conf_list):
        # assert confidence shape
        assert len(conf_list[0]) == self.attention_num
        
        data = torch.squeeze(
            torch.stack(data_list), 1
        )
        confidence = torch.from_numpy(np.array(conf_list)).float()
        self.priority_false_data = self.concatenate(data, self.priority_false_data)
        self.priority_false_confidence = self.concatenate(confidence, self.priority_false_confidence)
        if len(self.priority_false_data) > self.list_max_size//2:
            self.priority_false_data = self.priority_false_data[:self.list_max_size//2]
            self.priority_false_confidence = self.priority_false_confidence[:self.list_max_size//2]
        self.priority_false_length = len(self.priority_false_data)
        self._set_batch_num()
        self.counter = 0
        self.shuffle()

    def shuffle(self):
        self.shuffled_indices_true = np.setdiff1d(range(self.true_length), self.validate_indicies_true)
        self.shuffled_indices_true = np.random.permutation(self.shuffled_indices_true)
        
        self.true_train_length = len(self.shuffled_indices_true)
        self.true_test_length = len(self.validate_indicies_true)

        self.shuffled_indices_false = np.setdiff1d(range(self.false_length), self.validate_indicies_false)
        self.shuffled_indices_false = np.random.permutation(self.shuffled_indices_false)

        self.false_train_length = len(self.shuffled_indices_false)
        self.false_test_length = len(self.validate_indicies_false)

        self.shuffled_indices_priority_false = np.setdiff1d(range(self.priority_false_length), self.validate_indicies_priority_false)
        self.shuffled_indices_priority_false = np.random.permutation(self.shuffled_indices_priority_false)

        self.priority_false_train_length = len(self.shuffled_indices_priority_false)
        self.priority_false_test_length = len(self.validate_indicies_priority_false)
        

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
            data, labels, confidence = self._unibatch()
            data = self.transform(data)
            return data, labels, confidence

        if self.priority_false_length > 0:
            normal_false = self._get_minibatch(
                self.false_index(True),
                self.false_data,
                self.data_batchsize // 2,
                self.shuffled_indices_false
            )
            normal_false_confidence = self._get_minibatch(
                self.false_index(True),
                self.false_confidence,
                self.data_batchsize // 2,
                self.shuffled_indices_false
            )
            # print(torch.max(normal_false[0]))
            priority_false = self._get_minibatch(
                self.priority_false_index(),
                self.priority_false_data,
                self.data_batchsize - self.data_batchsize//2,
                self.shuffled_indices_priority_false
            )
            priority_false_confidence = self._get_minibatch(
                self.priority_false_index(),
                self.priority_false_confidence,
                self.data_batchsize - self.data_batchsize//2,
                self.shuffled_indices_priority_false
            )
            # print(torch.max(priority_false[0]))
            false_batch = self.concatenate(normal_false, priority_false)
            false_confidence_batch = self.concatenate(normal_false_confidence, priority_false_confidence)
        else:
            false_batch = self._get_minibatch(
                self.false_index(False),
                self.false_data,
                self.data_batchsize,
                self.shuffled_indices_false)
            false_confidence_batch = self._get_minibatch(
                self.false_index(False),
                self.false_confidence,
                self.data_batchsize,
                self.shuffled_indices_false)
        true_batch = self._get_minibatch(
            self.true_index(),
            self.true_data, 
            self.data_batchsize,
            self.shuffled_indices_true)
        true_confidence_batch = self._get_minibatch(
            self.true_index(),
            self.true_confidence, 
            self.data_batchsize,
            self.shuffled_indices_true)
        # print(torch.max(true_batch[0]))

        labels = [0]*len(false_batch) + [1]*len(true_batch)
        labels = torch.from_numpy(np.array(labels))

        self.counter += 1

        data = self.concatenate(false_batch, true_batch)
        confidence = self.concatenate(false_confidence_batch, true_confidence_batch)

        if shuffle_batch is True:
            shuffle_idxs = torch.randperm(len(data))
            data = data[shuffle_idxs]
            labels = labels[shuffle_idxs]
            confidence = confidence[shuffle_idxs]
            
        data = self.transform(data)
        
        if self.validate:
            if self.priority_false_length > 0:
                normal_false_val = self._get_minibatch(
                    self.false_val_index(True),
                    self.false_data,
                    self.data_batchsize//2,
                    self.validate_indicies_false
                )
                normal_false_confidence_val = self._get_minibatch(
                    self.false_val_index(True),
                    self.false_confidence,
                    self.data_batchsize//2,
                    self.validate_indicies_false
                )
                
                priority_false_val = self._get_minibatch(
                    self.priority_false_val_index(),
                    self.priority_false_data,
                    self.data_batchsize - self.data_batchsize//2,
                    self.validate_indicies_priority_false
                )
                priority_false_confidence_val = self._get_minibatch(
                    self.priority_false_val_index(),
                    self.priority_false_confidence,
                    self.data_batchsize - self.data_batchsize//2,
                    self.validate_indicies_priority_false
                )
    
                false_batch_val = self.concatenate(normal_false_val, priority_false_val)
                false_confidence_batch_val = self.concatenate(normal_false_confidence_val, priority_false_confidence_val)
            else:
                false_batch_val = self._get_minibatch(
                    self.false_val_index(False),
                    self.false_data,
                    self.data_batchsize,
                    self.shuffled_indices_false
                )
                false_confidence_batch_val = self._get_minibatch(
                    self.false_val_index(False),
                    self.false_confidence,
                    self.data_batchsize,
                    self.shuffled_indices_false
                )
            true_batch_val = self._get_minibatch(
                self.true_val_index(),
                self.true_data,
                self.data_batchsize,
                self.validate_indicies_true
            )
            true_confidence_batch_val = self._get_minibatch(
                self.true_val_index(),
                self.true_confidence,
                self.data_batchsize,
                self.validate_indicies_true
            )

            data_val = self.concatenate(false_batch_val, true_batch_val)
            labels_val = [0]*len(false_batch_val)+[1]*len(true_batch_val)
            labels_val = torch.from_numpy(np.array(labels_val))
            confidence_val = self.concatenate(false_confidence_batch_val, true_confidence_batch_val)

            data_val = self.transform(data_val)

            return data, labels, confidence, data_val, labels_val, confidence_val
        
        return data, labels, confidence

    def true_index(self):
        return (self.counter*self.data_batchsize) % self.true_train_length

    def true_val_index(self):
        return (self.counter*self.data_batchsize) % self.true_test_length
    
    def false_index(self, use_priority_false):
        if use_priority_false:
            return (self.counter*self.data_batchsize//2) % self.false_train_length
        return (self.counter*self.data_batchsize) % self.false_train_length

    def false_val_index(self, use_priority_false):
        if use_priority_false:
            return (self.counter*self.data_batchsize//2) % self.false_test_length
        return (self.counter*self.data_batchsize) % self.false_test_length

    def priority_false_index(self):
        return (self.counter*(self.data_batchsize - self.data_batchsize//2)) % self.priority_false_train_length

    def priority_false_val_index(self):
        return (self.counter*(self.data_batchsize - self.data_batchsize//2)) % self.priority_false_test_length


    def _unibatch(self):
        if self.true_length == 0:
            data =  self._get_minibatch(
                self.false_index(False), 
                self.false_data, 
                self.batchsize,
                self.shuffled_indices_false
                ) 
            confidence = self._get_minibatch(
                self.false_index(False), 
                self.false_confidence, 
                self.batchsize,
                self.shuffled_indices_false
                )
            labels = torch.from_numpy(np.array([0]*len(data)))
        else:
            data = self._get_minibatch(
                self.true_index(), 
                self.true_data, 
                self.batchsize,
                self.shuffled_indices_true
                )
            confidence = self._get_minibatch(
                self.true_index(), 
                self.true_confidence, 
                self.batchsize,
                self.shuffled_indices_true
                )
            labels = torch.from_numpy(np.array([1]*len(data)))
        
        return data, labels, confidence

    @staticmethod
    def concatenate(arr1, arr2):

        if len(arr1) == 0:
            return arr2
        else:
            return torch.cat((arr1, arr2), axis=0)
    