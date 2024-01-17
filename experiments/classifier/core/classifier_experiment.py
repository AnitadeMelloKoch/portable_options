import datetime
import random
import os
from re import I
import gin
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyparsing import List
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
                 num_testsets,
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
            self.use_gpu = True
            self.embedding = AutoEncoder().to("cuda")
        else:
            self.use_gpu = False
            self.embedding = AutoEncoder()

        self.embedding_loaded = False
        self.classifier = AttentionSet(use_gpu=True,
                                       vote_threshold=0.4,
                                       embedding=self.embedding,
                                       log_dir=self.log_dir)
        self.classifier_train_epochs = classifier_train_epochs
        
        self.test_sets = []
        for _ in range(num_testsets):
            self.test_sets.append(SetDataset(attention_num=self.classifier.attention_num))

        # full dataset (train + test). TODO: write fine_tune on full set, and retrain new classifier completely
        #self.full_set = SetDataset(attention_num=self.classifier.attention_num)

    def add_train_data(self, true_data_files, false_data_files):
        # add data to attention set
        self.classifier.add_data_from_files(true_data_files, 
                                            false_data_files)
        logging.info(f'ADDING TRAIN & VAL DATA...')
        logging.info(
            "Added {} true files and {} false files to classifier training set.".format(
                len(true_data_files), len(false_data_files))
            )
        logging.info(
            "TRAIN size: {} true images and {} false images".format(
                len(self.classifier.dataset.true_data), len(self.classifier.dataset.false_data))
            )
        logging.info(
            "VAL size: {} true images and {} false images".format(
                len(self.classifier.dataset.validate_indicies_true), len(self.classifier.dataset.validate_indicies_false))
            )
        
        '''self.full_set.add_true_files(true_data_files)
        self.full_set.add_false_files(false_data_files)
        logging.info(
            "Full dataset size: {} true images and {} false images".format(
                len(self.full_set.true_data), len(self.full_set.false_data))
            )'''
        

    def add_test_data(self, test_set_idx, true_data_files, false_data_files):
        # add data to the given test set
        assert isinstance(true_data_files, list)
        assert isinstance(false_data_files, list)

        test_set = self.test_sets[test_set_idx]
        
        test_set.add_true_files(true_data_files)
        test_set.add_false_files(false_data_files)
        logging.info(f'ADDING TEST DATA')
        logging.info(
            "Test dataset {}: added {} true files and {} false files.".format(
                test_set_idx, len(true_data_files), len(false_data_files))
            )
        logging.info(
            "  --num images: {} true images and {} false images.".format(
                len(test_set.true_data), len(test_set.false_data))
            )

        #self.full_set.add_true_files(true_data_files)
        #self.full_set.add_false_files(false_data_files)
        

    def train(self, save_ftr_dist=True):
        # train attention set
        self.classifier.train(epochs = self.classifier_train_epochs, 
                              save_ftr_distribution = save_ftr_dist)


    def stats_experiment(self):
        # get stats for training dataset: 
        train_stats = self.classifier.get_ftr_distribution()
        mean, sd, n = train_stats
        
        # get stats for "new" states: calculate sample confidence
        test_positive_confidence = self.classifier.sample_confidence(self.test_sets[0].true_data)
        test_negative_confidence = self.classifier.sample_confidence(self.test_sets[0].false_data)
        
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


    def predict(self, x): 
        with torch.no_grad():
            # init full_predictions tensor
            _,_,first_votes,_ = self.classifier.vote(x[0])
            full_predictions = torch.zeros((len(x), len(first_votes)))
            vote_predictions = torch.zeros((len(x)))
            if self.use_gpu:
                full_predictions = full_predictions.to("cuda")
                vote_predictions = vote_predictions.to("cuda")

            # fill full_predictions tensor
            for i in range(len(x)):
                this_vote,_,this_votes,_ = self.classifier.vote(x[i])
                full_predictions[i] = this_votes
                vote_predictions[i] = this_vote
            
        return full_predictions, vote_predictions

    def val_accuracy(self, val_true, val_false):
        with torch.no_grad():
            val_data = torch.cat((val_true, val_false))
            labels_true = torch.cat((torch.ones(len(val_true)), torch.zeros(len(val_false))))
            
            if self.use_gpu:
                val_data = val_data.to("cuda")
                labels_true = labels_true.to("cuda")

            labels_pred, vote_pred = self.predict(val_data) # have shape (num_samples, num_heads), want accu for each head 
            logging.info('===VAL ACCURACY (FULL)===')           
            full_acc = self.pred_accuracy(labels_pred, labels_true)
            logging.info('===VAL ACCURACY (VOTE)===')       
            vote_acc = self.pred_accuracy(vote_pred, labels_true)
            #print("Val accuracy for each attention: ", full_acc)
            #print("Val accuracy for voted prediction: ", vote_acc)
        return full_acc, vote_acc

    def pred_accuracy(self, labels_pred, labels_true):
        # pred labels can be (num_samples, num_heads)
        # true labels can be (num_samples)
        with torch.no_grad():

            if labels_pred.dim() > 1: 
                labels_true_expanded = labels_true.unsqueeze(1).expand_as(labels_pred)
                
                TP = ((labels_pred == 1) & (labels_true_expanded == 1)).float().sum(dim=0)
                FP = ((labels_pred == 1) & (labels_true_expanded == 0)).float().sum(dim=0)
                TN = ((labels_pred == 0) & (labels_true_expanded == 0)).float().sum(dim=0)
                FN = ((labels_pred == 0) & (labels_true_expanded == 1)).float().sum(dim=0)
                
                acc = (labels_pred == labels_true_expanded).float().mean(dim=0)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                
                logging.info(f'Accuracy: {acc} \nPrecision: {precision} \nRecall: {recall}')
            else:
                labels_pred = labels_pred.squeeze()
                
                TP = ((labels_pred == 1) & (labels_true == 1)).float().sum()
                FP = ((labels_pred == 1) & (labels_true == 0)).float().sum()
                TN = ((labels_pred == 0) & (labels_true == 0)).float().sum()
                FN = ((labels_pred == 0) & (labels_true == 1)).float().sum()
                
                acc = (labels_pred == labels_true).float().mean()
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                
                logging.info(f'\nAccuracy: {acc} \nPrecision: {precision} \nRecall: {recall}')
        return acc
    
    def add_unlabeled_data(self, unlabeled_test_set):
        # generate labels for the given test set
        # then use data w/ generated lebel to fine-tunes the classifier 
        # NOTE: fine tune w/ full dataset (train + test generated labels)
        with torch.no_grad():
            labels_pred, vote_pred = self.predict(unlabeled_test_set)
            vote_pred = vote_pred.to("cpu").numpy()
            #test_true = list(unlabeled_test_set[vote_pred == 1])
            #test_false = list(unlabeled_test_set[vote_pred == 0])
            test_true = [torch.from_numpy(img).float() for img in unlabeled_test_set[vote_pred == 1]]
            test_false = [torch.from_numpy(img).float() for img in unlabeled_test_set[vote_pred == 0]]
            
            self.classifier.add_data(test_true, test_false)
            #self.full_set.add_true_files(test_true)
            #self.full_set.add_false_files(test_false)
        
        
    def progressive_train(self, unlabeled_test_set, labels_actual):
        # continue training/finetuning by generating labels and use for train
        if self.use_gpu:
            labels_actual = labels_actual.to("cuda")
            
        self.add_unlabeled_data(unlabeled_test_set)

        with torch.no_grad():
            labels_pred, vote_pred = self.predict(unlabeled_test_set)
            logging.info('===TEST METRICS, BEFORE TRAIN ===')
            logging.info('FULL')
            self.pred_accuracy(labels_pred, labels_actual)
            logging.info('VOTE')
            self.pred_accuracy(vote_pred, labels_actual)

        # NOTE: retrain keeps updating ftr distribution (which will be used to compute next sample confidence)
        #self.train() 
        self.train(False) # TODO: not just use sample confidence, but also use pred probability to measure Confidence.

        with torch.no_grad():
            labels_pred, vote_pred = self.predict(unlabeled_test_set)
            logging.info(f'===TEST METRICS, AFTER TRAIN ===')
            logging.info('FULL')
            self.pred_accuracy(labels_pred, labels_actual)
            logging.info('VOTE')
            self.pred_accuracy(vote_pred, labels_actual)


    def progressive_train_testing(self, test_pos, test_neg, unlabeled_test_set, labels_actual):
        # continue training/finetuning by generating labels and use for train
        if self.use_gpu:
            labels_actual = torch.from_numpy(labels_actual)
            labels_actual = labels_actual.to("cuda")
        self.add_train_data(test_pos, test_neg)

        with torch.no_grad():
            labels_pred, vote_pred = self.predict(unlabeled_test_set)
            logging.info('===TEST METRICS, BEFORE TRAIN ===')
            logging.info('FULL')
            self.pred_accuracy(labels_pred, labels_actual)
            logging.info('VOTE')
            self.pred_accuracy(vote_pred, labels_actual)
            
        # NOTE: retrain keeps updating ftr distribution (which will be used to compute next sample confidence)
        #self.train() 
        self.train(False)

        with torch.no_grad():
            labels_pred, vote_pred = self.predict(unlabeled_test_set)
            logging.info(f'===TEST METRICS, AFTER TRAIN ===')
            logging.info('FULL')
            self.pred_accuracy(labels_pred, labels_actual)
            logging.info('VOTE')
            self.pred_accuracy(vote_pred, labels_actual)


    def retrain(self):
        # reset weights and train 
        #self.classifier.reset_weights() #TODO: reset weights function
        pass


        