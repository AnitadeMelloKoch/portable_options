import copy
import datetime
import logging
import os
import pickle
import random
from collections import deque
import re

import gin
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ray import train

from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.option.memory import SetDataset
from portable.utils.utils import set_seed
from evaluation.evaluators import DivDisEvaluatorClassifier




@gin.configurable
class MonteDivDisHyperparamSearchExperiment():
    def __init__(self,
                 experiment_name,
                 base_dir,
                 use_gpu):

        
        self.experiment_name = experiment_name
        self.use_gpu = use_gpu
        
        self.base_dir = os.path.join(base_dir, experiment_name)
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)



    
    def test_terminations(self,
                          dataset_positive,
                          dataset_negative,
                          classifier):
        
        counter = 0
        accuracy = np.zeros(classifier.head_num)
        accuracy_pos = np.zeros(classifier.head_num)
        accuracy_neg = np.zeros(classifier.head_num)
        
        for _ in range(dataset_positive.num_batches):
            counter += 1
            x, y = dataset_positive.get_batch()
            pred_y, votes = classifier.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_pos[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
        
        accuracy_pos /= counter
        
        total_count = counter
        counter = 0
        
        for _ in range(dataset_negative.num_batches):
            counter += 1
            x, y = dataset_negative.get_batch()
            pred_y, votes = classifier.predict(x)
            pred_y = pred_y.cpu()
            
            for idx in range(classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_neg[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
        
        accuracy_neg /= counter
        total_count += counter
        
        accuracy /= total_count
        
        weighted_acc = (accuracy_pos + accuracy_neg)/2
        
        # accuracy: accuracy over entire dataset 
        # weighted_acc: accuracy of pos&neg weighted equally
        return accuracy_pos, accuracy_neg, accuracy, weighted_acc



    def save_results_dict(self,
                          save_dict,
                          file_name):
        
        with open(os.path.join(self.log_dir, file_name), 'wb') as f:
            pickle.dump(save_dict, f)
        
    
    def plot(self,
             plot_file,
             x_values,
             accuracies,
             avg_accuracies,
             losses,
             
             plot_title,
             x_label,

             categorical=False,
             log_scale=False):

        losses = np.array(losses)
        accuracies = np.array(accuracies)
        avg_accuracies = np.array(avg_accuracies)
        
        ax_titles = ['Loss', 'Accuracy']
        y_labels = ['Final Training Loss', 'Test Accuracy']
    
        if not categorical: # most cases, line plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            print(accuracies.shape)
            print(accuracies)
            axes[0].plot(x_values, losses.mean(axis=1))
            axes[0].fill_between(x_values,
                                losses.mean(axis=1)-losses.std(axis=1),
                                losses.mean(axis=1)+losses.std(axis=1),
                                alpha=0.2)
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel(y_labels[0])
            axes[0].title.set_text(ax_titles[0])

            axes[1].plot(x_values, accuracies.mean(axis=1))
            axes[1].fill_between(x_values,
                                accuracies.mean(axis=1)-accuracies.std(axis=1),
                                accuracies.mean(axis=1)+accuracies.std(axis=1),
                                alpha=0.2)
            axes[1].plot(x_values, avg_accuracies.mean(axis=1))
            axes[1].fill_between(x_values,
                                avg_accuracies.mean(axis=1)-avg_accuracies.std(axis=1),
                                avg_accuracies.mean(axis=1)+avg_accuracies.std(axis=1),
                                alpha=0.2)
            axes[1].set_xlabel(x_label)
            axes[1].set_ylabel(y_labels[1])
            axes[1].title.set_text(ax_titles[1])


            if log_scale:
                axes[0].set_xscale('log')
                axes[1].set_xscale('log')
            
        else: # bar plot for categorical data
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            x_axis_data = np.arange(len(x_values))
            x_ticks = x_values
            bar_width = 0.5

            print(avg_accuracies.shape)
            print(avg_accuracies)
            
            axes[0].bar(x_axis_data, losses.mean(1), yerr=losses.std(1), align='center', alpha=0.8, ecolor='#3388EE', capsize=10)
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel(y_labels[0])
            axes[0].title.set_text(ax_titles[0])
            axes[0].set_xticks(x_axis_data)
            axes[0].set_xticklabels(x_ticks, rotation=35, ha='right')
            for label in axes[0].get_xticklabels():
                label.set_position((label.get_position()[0] + 0.05, label.get_position()[1]))
            
            axes[1].bar(x_axis_data-bar_width/2, accuracies.mean(1), yerr=accuracies.std(1), width=bar_width, align='center', alpha=0.8, ecolor='#3388EE', capsize=10)
            axes[1].bar(x_axis_data+bar_width/2, avg_accuracies.mean(1), yerr=avg_accuracies.std(1), width=bar_width, align='center', alpha=0.8, ecolor='#33DD99', capsize=10)
            axes[1].set_xlabel(x_label)
            axes[1].set_ylabel(y_labels[1])
            axes[1].title.set_text(ax_titles[1])
            axes[1].set_xticks(x_axis_data)
            axes[1].set_xticklabels(x_ticks, rotation=35, ha='right')
            for label in axes[1].get_xticklabels():
                label.set_position((label.get_position()[0] + 0.05, label.get_position()[1]))


        fig.suptitle(plot_title)
        fig.tight_layout()
        
        fig.savefig(plot_file)
        plt.close(fig)

    
    def train_classifier(self,
                         config,
                         
                         train_dataset,
                         positive_train_files,
                         negative_train_files,
                         unlabelled_train_files,
                            
                         room_list,
                         unlabelled_list,
                         
                         test_dataset_positive,
                         test_dataset_negative,
                         uncertain_dataset,
                         verbose=True):
        random_seed = np.random.randint(0, 100000)
        set_seed(random_seed)

        
        classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                    log_dir=self.log_dir,
                                    num_classes=2,
                                    diversity_weight=config['div_weight'],
                                    head_num=config['num_heads'],
                                    learning_rate=config['lr'],
                                    l2_reg_weight=config['l2_reg'],
                                    unlabelled_dataset_batchsize=config['unlabelled_batch_size'],)
        

        train_dataset.reset()

        if config['unlabelled_batch_size'] is not None:
            train_dataset.dynamic_unlabelled_batchsize = False
            train_dataset.unlabelled_batchsize = config['unlabelled_batch_size']
        else:
            train_dataset.dynamic_unlabelled_batchsize = True
            train_dataset.unlabelled_batchsize = 0
        
        train_dataset.add_true_files(positive_train_files)
        train_dataset.add_false_files(negative_train_files)
        train_dataset.add_unlabelled_files(unlabelled_train_files)

        classifier.dataset = train_dataset
        
        classifier.train(epochs=config['initial_epochs'])

        for i in range(len(room_list)):
            cur_room_unlab = unlabelled_list[i]
            cur_room_unlab = [np.load(file) for file in cur_room_unlab]
            cur_room_unlab = [img for list in cur_room_unlab for img in list]
            cur_room_unlab = [torch.from_numpy(img).float().squeeze() for img in cur_room_unlab]
            
            classifier.dataset.add_unlabelled_data(cur_room_unlab)
            train_loss = classifier.train(epochs=config['epochs_per_room'])

        
        accuracy_pos, accuracy_neg, accuracy, weighted_acc = self.test_terminations(test_dataset_positive, test_dataset_negative, classifier)
        best_weight_acc = max(weighted_acc)
        best_acc = accuracy[np.argmax(weighted_acc)]
        accuracy_pos = accuracy_pos[np.argmax(weighted_acc)]
        accuracy_neg = accuracy_neg[np.argmax(weighted_acc)]
        uncertain_pos_rate, _, _, _ = self.test_terminations(uncertain_dataset, uncertain_dataset, classifier)
        uncertain_pos_rate = uncertain_pos_rate[np.argmax(weighted_acc)]
        
        train.report({
            #"accuracy": accuracy, "weighted_acc": weighted_acc, 
            "best_weighted_acc": best_weight_acc, "best_acc": best_acc, 
            "positive_accuracy": accuracy_pos, "negative_accuracy": accuracy_neg,
            "uncertain_pos_rate": uncertain_pos_rate,
            "average_accuracy": np.mean(accuracy), "average_weighted_acc": np.mean(weighted_acc),
            "num_heads": config['num_heads'], "loss": train_loss})
    