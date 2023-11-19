import random
import os
import gin
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from portable.option.sets.attention_set import AttentionSet
from portable.option.memory import SetDataset
from portable.option.ensemble.custom_attention import AutoEncoder


logger = logging.getLogger(__name__)

@gin.configurable
class ClassifierExperiment():
    def __init__(self,
                 base_dir,
                 seed,
                 experiment_name,
                 classifier_train_epochs,
                 use_gpu=True):
        
        random.seed(seed)
        self.seed = seed
        
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, self.name, str(self.seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        #self.plot_dir = os.path.join(self.base_dir, 'plots')
        #self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        
        os.makedirs(self.log_dir, exist_ok=True)
        #os.makedirs(self.plot_dir, exist_ok=True)
        #os.makedirs(self.save_dir, exist_ok=True)
        
        if use_gpu:
            self.embedding = AutoEncoder().to("cuda")
        else:
            self.embedding = AutoEncoder()

        self.embedding_loaded = False
        self.classifier = AttentionSet(use_gpu=True,
                                       vote_threshold=0.4,
                                       embedding=self.embedding,
                                       log_dir=self.log_dir)
        self.classifier_train_epochs = classifier_train_epochs
        self.test_dataset = SetDataset()

    def add_train_data(self, true_data_files, false_data_files):
        # add data to attention set
        self.classifier.add_data_from_files(true_data_files, 
                                            false_data_files)

    def add_test_data(self, true_data_files, false_data_files):
        # add data to test set
        assert isinstance(true_data_files, list)
        assert isinstance(false_data_files, list)

        self.test_dataset.add_true_files(true_data_files)
        self.test_dataset.add_false_files(false_data_files)

    def train(self):
        # train attention set
        self.classifier.train(epochs = self.classifier_train_epochs, 
                              save_ftr_distribution = True)
        # 
        

    def stats_experiment(self):
        # get stats for training dataset: 
        train_stats = self.classifier.get_saved_ftr_distrbution()
        # get stats for "new" states: calculate sample confidence
        test_positive_stats = self.classifier.sample_confidence(self.test_dataset.true_data)
        test_negative_stats = self.classifier.sample_confidence(self.test_dataset.false_data)
        
        logger.info("training stats (mean, sd, n) =", train_stats)
        logger.info("test positive stats (avg sd away) =", test_positive_stats)
        logger.info("test negative stats (avg sd away) =", test_negative_stats)

        print("training stats=", train_stats)
        print("test (positive) stats=", test_positive_stats)
        print("test (negative) stats=", test_negative_stats)

    #future
    def classifier_weighted_experiment(self):
        pass