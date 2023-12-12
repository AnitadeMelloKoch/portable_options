import datetime
import random
import os
import gin
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

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
        
        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        
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
        logging.info(
            "Added {} true files and {} false files to attention set.".format(
                len(true_data_files), len(false_data_files))
            )
        logging.info(
            "Actual attention set dataset size: {} true files and {} false files".format(
                len(self.classifier.dataset.true_data), len(self.classifier.dataset.false_data))
            )

    def add_test_data(self, true_data_files, false_data_files):
        # add data to test set
        assert isinstance(true_data_files, list)
        assert isinstance(false_data_files, list)

        self.test_dataset.add_true_files(true_data_files)
        self.test_dataset.add_false_files(false_data_files)
        logging.info(
            "Added {} true files and {} false files to test dataset.".format(
                len(true_data_files), len(false_data_files))
            )
        logging.info(
            "Actual test set size: {} true files and {} false files".format(
                len(self.test_dataset.true_data), len(self.test_dataset.false_data))
            )

    def train(self):
        # train attention set
        self.classifier.train(epochs = self.classifier_train_epochs, 
                              save_ftr_distribution = True)

    def stats_experiment(self):
        # get stats for training dataset: 
        train_stats = self.classifier.get_ftr_distribution()
        mean, sd, n = train_stats
        
        # get stats for "new" states: calculate sample confidence
        test_positive_confidence = self.classifier.sample_confidence(self.test_dataset.true_data)
        test_negative_confidence = self.classifier.sample_confidence(self.test_dataset.false_data)
        
        logging.info("***TRAINING FEATURE STATS***")
        logging.info('ftr mean shape: ' + str(mean.shape))
        logging.info('ftr sd shape: ' + str(sd.shape))
        logging.info('ftr n shape: ' + str(n.shape))
        logging.info('ftr mean: ' + str(mean))
        logging.info('ftr sd: ' + str(sd))
        logging.info('ftr n: ' + str(n))
        logging.info("====================================")
        logging.info("***TEST POSITIVE CONFIDENCE STATS***")
        logging.info('test positive confidence shape: ' + str(test_positive_confidence.shape))
        logging.info('test positive confidence: ' + str(test_positive_confidence))
        logging.info('test positive confidence mean: ' + str(torch.mean(test_positive_confidence, dim=0)))
        logging.info('test positive confidence sd: ' + str(torch.std(test_positive_confidence, dim=0)))
        logging.info('test positive confidence n: ' + str(len(test_positive_confidence)))
        logging.info("====================================")
        logging.info("***TEST NEGATIVE CONFIDENCE STATS***")
        logging.info('test negative confidence shape: ' + str(test_negative_confidence.shape))
        logging.info('test negative confidence: ' + str(test_negative_confidence))
        logging.info('test negative confidence mean: ' + str(torch.mean(test_negative_confidence, dim=0)))
        logging.info('test negative confidence sd: ' + str(torch.std(test_negative_confidence, dim=0)))
        logging.info('test negative confidence n: ' + str(len(test_negative_confidence)))

    #future
    def classifier_weighted_experiment(self):
        pass