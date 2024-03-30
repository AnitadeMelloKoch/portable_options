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
from evaluation.evaluators import DivDisEvaluatorClassifier




@gin.configurable
class AdvancedMinigridDivDisSweepExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 use_gpu,
                 default_epochs,
                 default_div_weight,
                 default_div_lr,
                 default_num_heads,
                 default_l2_reg_weight,
                 
                 train_positive_files,
                 train_negative_files,
                 unlabelled_files,
                 test_positive_files,
                 test_negative_files,
                 
                 evaluation_sample_size):

        
        self.experiment_name = experiment_name
        self.use_gpu = use_gpu
        
        self.base_dir = os.path.join(base_dir, experiment_name)
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.default_epochs = default_epochs
        self.default_div_weight = default_div_weight
        self.default_div_lr = default_div_lr
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
            pred_y = classifier.predict(x)
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
            pred_y = classifier.predict(x)
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
        evaluator = DivDisEvaluatorClassifier(classifier, image_input=True, batch_size=32, base_dir=self.base_dir)
        evaluator.add_test_files(self.test_positive_files, self.test_negative_files)
        #evaluator.test_dataset.set_transform_function(transform)
        evaluator.evaluate(test_sample_size=self.evaluation_sample_size)
        return evaluator.get_head_complexity()


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
             complexities,
             
             plot_title,
             x_label,

             categorical=False,
             log_scale=False):

        losses = np.array(losses)
        accuracies = np.array(accuracies)
        avg_accuracies = np.array(avg_accuracies)
        complexities = np.array(complexities)
        
        ax_titles = ['Loss', 'Accuracy', 'Complexity']
        y_labels = ['Final Training Loss', 'Test Accuracy', 'Head Complexity']
    
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

            axes[2].plot(x_values, complexities.mean(axis=1))
            axes[2].fill_between(x_values,
                                np.maximum(complexities.mean(axis=1)-complexities.std(axis=1),0),
                                complexities.mean(axis=1)+complexities.std(axis=1),
                                alpha=0.2)
            axes[2].set_xlabel(x_label)
            axes[2].set_ylabel(y_labels[2])
            axes[2].title.set_text(ax_titles[2])
            if log_scale:
                axes[0].set_xscale('log')
                axes[1].set_xscale('log')
                axes[2].set_xscale('log')
            
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
            
            axes[1].bar(x_axis_data-bar_width/2, accuracies.mean(1), yerr=accuracies.std(1), width=bar_width, align='center', alpha=0.8, ecolor='#3388EE', capsize=10)
            axes[1].bar(x_axis_data+bar_width/2, avg_accuracies.mean(1), yerr=avg_accuracies.std(1), width=bar_width, align='center', alpha=0.8, ecolor='#33DD99', capsize=10)
            axes[1].set_xlabel(x_label)
            axes[1].set_ylabel(y_labels[1])
            axes[1].title.set_text(ax_titles[1])
            axes[1].set_xticks(x_axis_data)
            axes[1].set_xticklabels(x_ticks, rotation=35, ha='right')

            axes[2].bar(x_axis_data, complexities.mean(1), yerr=complexities.std(1), align='center', alpha=0.8, ecolor='#3388EE', capsize=10)
            axes[2].set_xlabel(x_label)
            axes[2].set_ylabel(y_labels[0])
            axes[2].title.set_text(ax_titles[0])
            axes[2].set_xticks(x_axis_data)
            axes[2].set_xticklabels(x_ticks, rotation=35, ha='right')

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
        results_comp = []

        
        weights = np.logspace(start_weight, end_weight, num_samples)
        
        for weight in tqdm(weights, desc="weights", position=0):
            results_weight.append(weight)
            weight_acc = []
            weight_avg_acc = []
            weight_loss = []
            weight_comp = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_div_lr,
                                              l2_reg_weight=self.default_l2_reg_weight)
                
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

                comps = self.head_complexity(classifier)
                weight_comp.append(comps[np.argmax(acc)])
        
            results_acc.append(weight_acc)
            results_avg_acc.append(weight_avg_acc)
            results_loss.append(weight_loss)
            results_comp.append(weight_comp)

        save_dict = {"weights": results_weight,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,
                     "complexities": results_comp}
        self.save_results_dict(save_dict,
                               "weights_sweep.pkl")


        self.plot(os.path.join(self.plot_dir, "div_weight_sweep.png"),
                  results_weight,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  results_comp,
                  "Sweep over Diversity Weight",
                  "Diversity Weight",
                  log_scale=True)
        

    
    def sweep_epochs(self,
                     start_epochs,
                     end_epochs,
                     step_size,
                     num_seeds):
        
        # 1D array
        results_epoch = []
        # 2D array for multiple seeds
        results_acc = []
        results_avg_acc = []
        results_loss = []
        results_comp = []

        
        
        for epochs in tqdm(range(start_epochs, end_epochs+1, step_size), desc="epochs", position=0):
            results_epoch.append(epochs)
            epoch_acc = []
            epoch_avg_acc = []
            epoch_loss = []
            epoch_comp = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_div_lr,
                                              l2_reg_weight=self.default_l2_reg_weight)
                
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

                comps = self.head_complexity(classifier)
                epoch_comp.append(comps[np.argmax(acc)])
            
            results_acc.append(epoch_acc)
            results_avg_acc.append(epoch_avg_acc)
            results_loss.append(epoch_loss)
            results_comp.append(epoch_comp)
        
        save_dict = {"epochs": results_epoch,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc, 
                     "losses": results_loss,
                     "complexities": results_comp}
        self.save_results_dict(save_dict,
                               "epochs_sweep.pkl")
                    

        self.plot(os.path.join(self.plot_dir, "epoch_sweep.png"),
                  results_epoch,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  results_comp,
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
        results_comp = []

        
        for num_heads in tqdm(range(start_size, end_size+1, step_size), desc="size", position=0):
            results_size.append(num_heads)
            size_acc = []
            size_avg_acc = []
            size_loss = []
            size_comp = []
            

            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=num_heads,
                                              learning_rate=self.default_div_lr,
                                              l2_reg_weight=self.default_l2_reg_weight)
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
                
                comps = self.head_complexity(classifier)
                size_comp.append(comps[np.argmax(acc)]) # complexity of best head

            results_acc.append(size_acc)
            results_avg_acc.append(size_avg_acc)
            results_loss.append(size_loss)
            results_comp.append(size_comp)
        
        save_dict = {"size": results_size,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc, 
                     "losses": results_loss,
                     "complexities": results_comp}
        self.save_results_dict(save_dict,
                               "size_sweep.pkl")

        
        self.plot(os.path.join(self.plot_dir, "ensemble_size_sweep.png"),
                  results_size,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  results_comp,
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
        results_comp = []

        
        for batch_size in tqdm(range(start_batchsize, end_batchsize+1, batch_stepsize), desc="batch_sizes", position=0):
            results_batchsize.append(batch_size)
            batch_acc = []
            batch_avg_acc = []
            batch_loss = []
            batch_comp = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_div_lr,
                                              unlabelled_dataset_batchsize=batch_size,
                                              l2_reg_weight=self.default_l2_reg_weight)
                
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

                comps = self.head_complexity(classifier)
                batch_comp.append(comps[np.argmax(acc)])
                
            results_acc.append(batch_acc)
            results_avg_acc.append(batch_avg_acc)
            results_loss.append(batch_loss)
            results_comp.append(batch_comp)
        
        save_dict = {"batchsizes": results_batchsize,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,
                     "complexities": results_comp}
        self.save_results_dict(save_dict,
                               "batch_sweep.pkl")
        

        self.plot(os.path.join(self.plot_dir, "batch_sweep.png"),
                  results_batchsize,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  results_comp,
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
        results_comp = []

        
        lrs = np.logspace(start_lr, end_lr, num_samples)
        
        for lr in tqdm(lrs, desc="lrs", position=0):
            results_lr.append(lr)
            lr_acc = []
            lr_avg_acc = []
            lr_loss = []
            lr_comp = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=lr,
                                              l2_reg_weight=self.default_l2_reg_weight)
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

                comps = self.head_complexity(classifier)
                lr_comp.append(comps[np.argmax(acc)])
                
            results_acc.append(lr_acc)
            results_avg_acc.append(lr_avg_acc)
            results_loss.append(lr_loss)
            results_comp.append(lr_comp)

        save_dict = {"learning_rates": results_lr,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,
                     "complexities": results_comp}
        
        self.save_results_dict(save_dict,
                               "lr_sweep.pkl")
                

        self.plot(os.path.join(self.plot_dir, "lr_sweep.png"),
                  results_lr,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  results_comp,
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
        results_comp = []

        
        l2_weights = np.logspace(start_l2, end_l2, num_samples)
        
        for reg_weight in tqdm(l2_weights, desc="l2_reg_weights", position=0):
            results_reg_weights.append(reg_weight)
            lr_acc = []
            lr_avg_acc = []
            lr_loss = []
            lr_comp = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_div_lr,
                                              l2_reg_weight=reg_weight)
                
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

                comps = self.head_complexity(classifier)
                lr_comp.append(comps[np.argmax(acc)])
                
            results_acc.append(lr_acc)
            results_avg_acc.append(lr_avg_acc)
            results_loss.append(lr_loss)
            results_comp.append(lr_comp)

        save_dict = {"reg_weights": results_reg_weights,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,
                     "complexities": results_comp}
        
        self.save_results_dict(save_dict,
                               "reg_weight_sweep.pkl")
                

        self.plot(os.path.join(self.plot_dir, "reg_weight_sweep.png"),
                  results_reg_weights,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  results_comp,
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
        results_comp = []
        
        for overlap in tqdm(np.linspace(start_ratio, end_ratio, step_size), desc="unlabelled_overlap_ratio", position=0):
            results_overlap.append(overlap)
            overlap_acc = []
            overlap_avg_acc = []
            overlap_loss = []
            overlap_comp = []
            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_div_lr,
                                              l2_reg_weight=self.default_l2_reg_weight)
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

                comps = self.head_complexity(classifier)
                overlap_comp.append(comps[np.argmax(acc)])
                 
            results_acc.append(overlap_acc)
            results_avg_acc.append(overlap_avg_acc)
            results_loss.append(overlap_loss)
            results_comp.append(overlap_comp)

        save_dict = {"ratio": results_overlap,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,
                     "complexities": results_comp}
        self.save_results_dict(save_dict,
                               "unlabelled_overlap_ratio_sweep.pkl")
                
        self.plot(os.path.join(self.plot_dir, "unlabelled_overlap_ratio_sweep.png"),
                  results_overlap,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  results_comp,
                  
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
        results_comp = []
        
        for variety in tqdm(range(len(all_combination_files)), desc="unlabelled_variety_combinations", position=0):
            results_variety.append(variety_combinations[variety])
            variety_acc = []
            variety_avg_acc = []
            variety_loss = []
            variety_comp = []

            
            for seed in tqdm(range(num_seeds), desc="seeds", position=1, leave=False):
                set_seed(seed)
                classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                              log_dir=self.log_dir,
                                              diversity_weight=self.default_div_weight,
                                              head_num=self.default_num_heads,
                                              learning_rate=self.default_div_lr,
                                              l2_reg_weight=self.default_l2_reg_weight)
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

                comps = self.head_complexity(classifier)
                variety_comp.append(comps[np.argmax(acc)])
                
            results_acc.append(variety_acc)
            results_avg_acc.append(variety_avg_acc)
            results_loss.append(variety_loss)
            results_comp.append(variety_comp)
        
        save_dict = {"variety": results_variety,
                     "variety_combinations": variety_combinations,
                     "accuracies": results_acc,
                     "avg_accuracies": results_avg_acc,
                     "losses": results_loss,
                     "complexities": results_comp,
                     }
        self.save_results_dict(save_dict,
                               "unlabelled_variety_sweep.pkl")

        self.plot(os.path.join(self.plot_dir, "unlabelled_variety_sweep.png"),
                  results_variety,
                  results_acc,
                  results_avg_acc,
                  results_loss,
                  results_comp,
                  "Sweep over Unlabelled Variety",
                  "Unlabelled Variety (seed, color, random state)",
                  categorical=True)

    
    
    