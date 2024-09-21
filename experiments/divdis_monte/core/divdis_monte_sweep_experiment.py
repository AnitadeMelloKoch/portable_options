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

from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.option.memory import SetDataset
from portable.utils.utils import set_seed
#from evaluation.evaluators import DivDisEvaluatorClassifier




@gin.configurable
class MonteDivDisSweepExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 use_gpu,
                 default_num_heads,
                 default_lr,
                 default_div_weight,
                 default_l2_reg_weight,
                 default_epochs,
                 
                 train_positive_files,
                 train_negative_files,
                 unlabelled_files,
                 test_positive_files,
                 test_negative_files,
                 
                 evaluation_sample_size=None,
                 seed=0):

        
        self.experiment_name = experiment_name
        self.use_gpu = use_gpu
        
        self.base_dir = os.path.join(base_dir, experiment_name)
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.default_epochs = default_epochs
        self.default_div_weight = default_div_weight
        self.default_lr = default_lr
        self.default_num_heads = default_num_heads
        self.default_l2_reg_weight = default_l2_reg_weight
        
        self.dataset_positive = SetDataset(max_size=1e6,
                                      batchsize=64)
        self.dataset_negative = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        #self.dataset_positive.set_transform_function(transform)
        #self.dataset_negative.set_transform_function(transform)
        
        self.dataset_positive.add_true_files(test_positive_files)
        self.dataset_negative.add_false_files(test_negative_files)

        self.test_positive_files = test_positive_files
        self.test_negative_files = test_negative_files
        
        self.train_positive_files = train_positive_files
        self.train_negative_files = train_negative_files
        self.unlabelled_files = unlabelled_files

        self.evaluation_sample_size = evaluation_sample_size

        self.seed = seed

    
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


    def head_complexity(self, classifier):
        #evaluator = DivDisEvaluatorClassifier(classifier, image_input=True, batch_size=32, base_dir=self.base_dir)
        #evaluator.add_test_files(self.test_positive_files, self.test_negative_files)
        ##evaluator.test_dataset.set_transform_function(transform)
        #evaluator.evaluate(test_sample_size=self.evaluation_sample_size)
        #return evaluator.get_head_complexity()
        return np.zeros(classifier.head_num)


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
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
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
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
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

    
    def sweep_class_div_weight(self,
                               start_weight,
                               end_weight,
                               num_samples,
                               num_seeds):
        
        # 1D array
        results_weight = []
        # 2D array for multiple seeds
        results_acc = []
        results_avg_acc = []
        results_loss = []

        
        weights = np.logspace(start_weight, end_weight, num_samples)
        
        for weight in tqdm(weights, desc="weights", position=0):
            results_weight.append(weight)
            weight_acc = []
            weight_avg_acc = []
            weight_loss = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(self.seed+seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              num_classes=2,
                                              diversity_weight=weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_lr,
                                              l2_reg_weight=self.default_l2_reg_weight,
                                              model_name="monte_cnn")
                
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=self.unlabelled_files)
                
                train_loss = classifier.train(self.default_epochs)
                weight_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                weight_acc.append(max(acc))
                weight_avg_acc.append(np.mean(acc))

        
            results_acc.append(weight_acc)
            results_avg_acc.append(weight_avg_acc)
            results_loss.append(weight_loss)

        save_dict = {"weights": results_weight,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,}
        self.save_results_dict(save_dict,
                               "weights_sweep.pkl")


        self.plot(os.path.join(self.plot_dir, "div_weight_sweep.png"),
                  results_weight,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  "Sweep over Diversity Weight",
                  "Diversity Weight",
                  log_scale=True)
        

    
    def sweep_epochs(self,
                     start_epochs,
                     end_epochs,
                     step_size,
                     num_seeds,
                     epochs_list=None):
        
        # 1D array
        results_epoch = []
        # 2D array for multiple seeds
        results_acc = []
        results_avg_acc = []
        results_loss = []

        if epochs_list is None:
            epochs_list = range(start_epochs, end_epochs+1, step_size)


        for epochs in tqdm(epochs_list, desc="epochs", position=0):
            results_epoch.append(epochs)
            epoch_acc = []
            epoch_avg_acc = []
            epoch_loss = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(self.seed+seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_lr,
                                              l2_reg_weight=self.default_l2_reg_weight,
                                              model_name="monte_cnn")
                
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=self.unlabelled_files)
                
                train_loss = classifier.train(epochs)
                epoch_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                
                epoch_acc.append(max(acc))
                epoch_avg_acc.append(np.mean(acc))
            
            results_acc.append(epoch_acc)
            results_avg_acc.append(epoch_avg_acc)
            results_loss.append(epoch_loss)
        
        save_dict = {"epochs": results_epoch,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc, 
                     "losses": results_loss,}
        self.save_results_dict(save_dict,
                               "epochs_sweep.pkl")
                    

        self.plot(os.path.join(self.plot_dir, "epoch_sweep.png"),
                  results_epoch,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  "Sweep over Train Epochs",
                  "Train Epochs")
        

    
    def sweep_ensemble_size(self,
                            start_size,
                            end_size,
                            step_size,
                            num_seeds):
        # 1D array
        results_size = []
        # 2D array for multiple seeds
        results_acc = []
        results_avg_acc = []
        results_loss = []

        
        for num_heads in tqdm(range(start_size, end_size+1, step_size), desc="size", position=0):
            results_size.append(num_heads)
            size_acc = []
            size_avg_acc = []
            size_loss = []
            

            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(self.seed+seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=num_heads,
                                              learning_rate=self.default_lr,
                                              l2_reg_weight=self.default_l2_reg_weight,
                                              model_name="monte_cnn")
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=self.unlabelled_files)
                
                train_loss = classifier.train(self.default_epochs)
                size_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                size_acc.append(max(acc))
                size_avg_acc.append(np.mean(acc))
                

            results_acc.append(size_acc)
            results_avg_acc.append(size_avg_acc)
            results_loss.append(size_loss)
        
        save_dict = {"size": results_size,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc, 
                     "losses": results_loss,}
        self.save_results_dict(save_dict,
                               "size_sweep.pkl")

        
        self.plot(os.path.join(self.plot_dir, "ensemble_size_sweep.png"),
                  results_size,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  "Sweep over Ensemble Size",
                  "No. Ensemble Members")

        

    def sweep_div_batch_size(self,
                             start_batchsize,
                             end_batchsize,
                             batch_stepsize,
                             num_seeds):
        # 1D array
        results_batchsize = []
        # 2D array for multiple seeds
        results_acc = []
        results_avg_acc = []
        results_loss = []

        
        for batch_size in tqdm(range(start_batchsize, end_batchsize+1, batch_stepsize), desc="batch_sizes", position=0):
            results_batchsize.append(batch_size)
            batch_acc = []
            batch_avg_acc = []
            batch_loss = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(self.seed+seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_lr,
                                              unlabelled_dataset_batchsize=batch_size,
                                              l2_reg_weight=self.default_l2_reg_weight,
                                              model_name="monte_cnn")
                
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=self.unlabelled_files)
                
                train_loss = classifier.train(self.default_epochs)
                batch_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                batch_acc.append(max(acc))
                batch_avg_acc.append(np.mean(acc))

                
            results_acc.append(batch_acc)
            results_avg_acc.append(batch_avg_acc)
            results_loss.append(batch_loss)
        
        save_dict = {"batchsizes": results_batchsize,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,}
        self.save_results_dict(save_dict,
                               "batch_sweep.pkl")
        

        self.plot(os.path.join(self.plot_dir, "batch_sweep.png"),
                  results_batchsize,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  "Sweep over Unlabelled Dataset Batchsize",
                  "Batchsize")
        

    
    def sweep_lr(self,
                 start_lr,
                 end_lr,
                 num_samples,
                 num_seeds):
        # 1D array
        results_lr = []
        # 2D array for multiple seeds
        results_acc = []
        results_avg_acc = []
        results_loss = []

        
        lrs = np.logspace(start_lr, end_lr, num_samples)
        
        for lr in tqdm(lrs, desc="lrs", position=0):
            results_lr.append(lr)
            lr_acc = []
            lr_avg_acc = []
            lr_loss = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(self.seed+seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=lr,
                                              l2_reg_weight=self.default_l2_reg_weight,
                                              model_name="monte_cnn")
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=self.unlabelled_files)
                
                train_loss = classifier.train(self.default_epochs)
                lr_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                lr_acc.append(max(acc))
                lr_avg_acc.append(np.mean(acc))

                
            results_acc.append(lr_acc)
            results_avg_acc.append(lr_avg_acc)
            results_loss.append(lr_loss)

        save_dict = {"learning_rates": results_lr,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,}
        
        self.save_results_dict(save_dict,
                               "lr_sweep.pkl")
                

        self.plot(os.path.join(self.plot_dir, "lr_sweep.png"),
                  results_lr,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  "Sweep over Learning Rate",
                  "Learning Rate",
                  log_scale=True)
    

    def sweep_l2_reg_weight(self,
                 start_l2,
                 end_l2,
                 num_samples,
                 num_seeds):
        # 1D array
        results_reg_weights = []
        # 2D array for multiple seeds
        results_acc = []
        results_avg_acc = []
        results_loss = []

        
        l2_weights = np.logspace(start_l2, end_l2, num_samples)
        
        for reg_weight in tqdm(l2_weights, desc="l2_reg_weights", position=0):
            results_reg_weights.append(reg_weight)
            lr_acc = []
            lr_avg_acc = []
            lr_loss = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(self.seed+seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_lr,
                                              l2_reg_weight=reg_weight,
                                              model_name="monte_cnn")
                
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=self.unlabelled_files)
                
                train_loss = classifier.train(self.default_epochs)
                lr_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                lr_acc.append(max(acc))
                lr_avg_acc.append(np.mean(acc))

                
            results_acc.append(lr_acc)
            results_avg_acc.append(lr_avg_acc)
            results_loss.append(lr_loss)

        save_dict = {"reg_weights": results_reg_weights,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,}
        
        self.save_results_dict(save_dict,
                               "reg_weight_sweep.pkl")
                

        self.plot(os.path.join(self.plot_dir, "reg_weight_sweep.png"),
                  results_reg_weights,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  "Sweep over L2 Regularization Weight",
                  "L2 Regularization Weight",
                  log_scale=True)
        


    def sweep_div_overlap(self,
                        start_ratio,
                        end_ratio,
                        step_size,
                        num_seeds):
        # 1D array
        results_overlap = []
        # 2D array for multiple seeds
        results_acc = []
        results_avg_acc = []
        results_loss = []
        
        for overlap in tqdm(np.linspace(start_ratio, end_ratio, step_size), desc="unlabelled_overlap_ratio", position=0):
            results_overlap.append(overlap)
            overlap_acc = []
            overlap_avg_acc = []
            overlap_loss = []
            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(self.seed+seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_lr,
                                              l2_reg_weight=self.default_l2_reg_weight,
                                              model_name="monte_cnn")
                unlabelled_num = len(self.unlabelled_files)
                unlabelled_files = random.sample(self.unlabelled_files, int(unlabelled_num*(1-overlap)))
                train_files = self.train_positive_files + self.train_negative_files
                train_files += train_files
                unlabelled_files += random.sample(train_files, int(unlabelled_num*overlap))
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=unlabelled_files)
                
                train_loss = classifier.train(self.default_epochs)
                overlap_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                overlap_acc.append(max(acc))
                overlap_avg_acc.append(np.mean(acc))

                 
            results_acc.append(overlap_acc)
            results_avg_acc.append(overlap_avg_acc)
            results_loss.append(overlap_loss)

        save_dict = {"ratio": results_overlap,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,}
        self.save_results_dict(save_dict,
                               "unlabelled_overlap_ratio_sweep.pkl")
                
        self.plot(os.path.join(self.plot_dir, "unlabelled_overlap_ratio_sweep.png"),
                  results_overlap,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  
                  "Sweep over Unlabelled Overlap Ratio",
                  "Unlabelled Overlap Ratio")

    
    def sweep_div_variety(self,
                          variety_combinations,
                          all_combination_files,
                          num_seeds):
        # 1D array
        results_variety = []
        # 2D array for multiple seeds
        results_acc = []
        results_avg_acc = []
        results_loss = []
        
        for variety in tqdm(range(len(all_combination_files)), desc="unlabelled_variety_combinations", position=0):
            results_variety.append(variety_combinations[variety])
            variety_acc = []
            variety_avg_acc = []
            variety_loss = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(self.seed+seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_lr,
                                              l2_reg_weight=self.default_l2_reg_weight,
                                              model_name="monte_cnn")
                classifier.add_data(positive_files=self.train_positive_files,
                                    negative_files=self.train_negative_files,
                                    unlabelled_files=all_combination_files[variety])
                
                train_loss = classifier.train(self.default_epochs)
                variety_loss.append(train_loss)
                
                _, _, _, acc = self.test_terminations(self.dataset_positive,
                                                      self.dataset_negative,
                                                      classifier)
                variety_acc.append(max(acc))
                variety_avg_acc.append(np.mean(acc))

                
            results_acc.append(variety_acc)
            results_avg_acc.append(variety_avg_acc)
            results_loss.append(variety_loss)
        
        save_dict = {"variety": results_variety,
                     "variety_combinations": variety_combinations,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,
                     }
        self.save_results_dict(save_dict,
                               "unlabelled_variety_sweep.pkl")

        self.plot(os.path.join(self.plot_dir, "unlabelled_variety_sweep.png"),
                  results_variety,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  "Sweep over Unlabelled Variety",
                  "Unlabelled Variety (seed, color, random state)",
                  categorical=True)

    
    

    def grid_search(self,
                    lr_range,
                    div_weight_range,
                    l2_reg_range,
                    head_num_range,
                    epochs_range,
                    num_seeds):
        
        # Arrays to store results for grid search
        results_head_num = []
        results_lr = []
        results_div_weight = []
        results_l2_reg_weight = []
        results_epochs = epochs_range
        
        # 2D array for multiple seeds
        results_acc = []
        results_avg_acc = []
        results_loss = []

        results_raw_acc = []
        results_positive_acc = []
        results_negative_acc = []
        
        # Create grid of hyperparameter combinations
        for head_num in tqdm(head_num_range, desc="head_nums", position=0, leave=False):
            for lr in tqdm(lr_range, desc="learning_rates", position=1, leave=False):
                for div_weight in tqdm(div_weight_range, desc="div_weights", position=2):
                    for l2_reg_weight in tqdm(l2_reg_range, desc="l2_reg_weights", position=3, leave=False):
                        for epochs in tqdm(epochs_range, desc="epochs", position=4, leave=False):
                            
                        
                            # Record the current hyperparameters being tested
                            results_head_num.append(head_num)
                            results_lr.append(lr)
                            results_div_weight.append(div_weight)
                            results_l2_reg_weight.append(l2_reg_weight)
                            results_epochs.append(epochs)

                            grid_acc = []
                            grid_avg_acc = []
                            grid_loss = []
                            grid_raw_acc = []
                            grid_positive_acc = []
                            grid_negative_acc = []

                            # Train with multiple seeds for robustness
                            for seed in tqdm(range(num_seeds), desc="seeds", position=5, leave=False):
                                set_seed(self.seed+seed)
                                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                                            log_dir=self.log_dir,
                                                            num_classes=2,
                                                            diversity_weight=div_weight,
                                                            head_num=head_num,
                                                            learning_rate=lr,
                                                            l2_reg_weight=l2_reg_weight,
                                                            model_name="monte_cnn")
                                classifier.add_data(positive_files=self.train_positive_files,
                                                    negative_files=self.train_negative_files,
                                                    unlabelled_files=self.unlabelled_files)
                                
                                train_loss = classifier.train(epochs)
                                grid_loss.append(train_loss)
                                
                                accuracy_pos, accuracy_neg, accuracy, weighted_acc = self.test_terminations(self.dataset_positive,
                                                                    self.dataset_negative,
                                                                    classifier)
                                best_idx = np.argmax(weighted_acc)
                                grid_acc.append(max(weighted_acc))
                                grid_avg_acc.append(np.mean(weighted_acc))
                                grid_raw_acc.append(accuracy[best_idx])
                                grid_positive_acc.append(accuracy_pos[best_idx])
                                grid_negative_acc.append(accuracy_neg[best_idx])


                            # Store results
                            results_acc.append(grid_acc)
                            results_avg_acc.append(grid_avg_acc)
                            results_loss.append(grid_loss)

                            results_raw_acc.append(grid_raw_acc)
                            results_positive_acc.append(grid_positive_acc)
                            results_negative_acc.append(grid_negative_acc)

        # Save the results
        save_dict = {"div_weights": results_div_weight,
                    "head_nums": results_head_num,
                    "learning_rates": results_lr,
                    "l2_reg_weights": results_l2_reg_weight,
                    "epochs": epochs_range,
                    "avg_best_weighted_acc": np.mean(results_acc, axis=1), # over seeds
                    "weighted_acc": results_acc,        # 2D (m,num_seeds)
                    "avg_accuracies": results_avg_acc,  # 2D
                    "losses": results_loss,
                    "raw_acc": results_raw_acc,         # 2D
                    "positive_acc": results_positive_acc, # 2D
                    "negative_acc": results_negative_acc, # 2D
                    }

        self.save_results_dict(save_dict,
                            "grid_search_results.pkl")

        results_weighted_acc = np.array(results_acc)
        print(results_weighted_acc.shape)
        print(results_weighted_acc)

        # Plot the results (you can customize the x-axis and what to compare)
        #self.plot(os.path.join(self.plot_dir, "grid_search.png"),
        #        results_div_weight,  # Example: Plotting against div_weight, you can customize this
        #        results_acc,
        #        results_avg_acc,
        #        results_loss,
        #        "Grid Search Results",
        #        "Diversity Weight")
