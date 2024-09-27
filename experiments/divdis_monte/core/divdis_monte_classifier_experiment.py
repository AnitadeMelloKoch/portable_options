import copy
import datetime
import logging
import os
import pickle
import random
import re

import gin
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from portable.option.divdis.divdis_classifier import DivDisClassifier
from portable.option.memory import SetDataset
from portable.utils.utils import set_seed



def save_image(img, save_dir, batch_number, img_number_within_batch):
            # Construct filename to include both batch number and image number within the batch
            filename = f"{save_dir}/batch_{batch_number}_image_{img_number_within_batch}.png"
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            for i, ax in enumerate(axes.flat):
                ax.imshow(img[i], cmap='gray')
            plt.tight_layout()
            plt.savefig(filename)
            plt.close(fig)

def worker_initializer():
    plt.switch_backend('Agg')

#def transform(x):
#    # only keep the last channel of imgs
#    x = x[:, -1]
#    x = x.unsqueeze(1)
#    return x

def get_sorted_filenames(directory):
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filenames.append(file)
    filenames.sort()
    
    return filenames


def get_monte_filenames(filenames, task, room, init_term_uncertain, pos_neg=None):
    """
    Filters filenames by task, room, status (initiation/termination/uncertain), and label (positive/negative).

    Parameters:
    - filenames (list of str): Available filenames in resources folder
    - task (str): Task name (e.g., "climb_down_ladder").
    - room (str): Room number (e.g., "2", "14left").
    - init_term_uncertain (str): 'initiation', 'termination', or 'uncertain'.
    - pos_neg (str, optional): 'positive' or 'negative', ignored if 'uncertain'.

    Returns:
    - list of str: Filenames matching the criteria or an empty list.
    """
    
    if init_term_uncertain == 'uncertain':
        # If uncertain, we don't care about positive/negative
        pattern = re.compile(rf"{task}_room{room}(_1)?_{init_term_uncertain}\.npy")
    else:
        # For initiation/termination, match both positive/negative if pos_neg is not None
        pattern = re.compile(rf"{task}_room{room}(_1)?_{init_term_uncertain}_{pos_neg}\.npy")
    
    return [filename for filename in filenames if pattern.match(filename)]




@gin.configurable 
class MonteDivDisClassifierExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 use_gpu,
                 
                 classifier_num_classes,
                 
                 classifier_head_num,
                 classifier_learning_rate,
                 classifier_diversity_weight,
                 classifier_l2_reg_weight,
                 classifier_initial_epochs,
                 classifier_per_room_epochs,
                 classifier_model_name,
                 ):
        
        self.seed = seed 
        self.base_dir = base_dir
        self.experiment_name = experiment_name
        self.use_gpu = use_gpu
        
        self.initial_epochs = classifier_initial_epochs
        self.per_room_epochs = classifier_per_room_epochs

        self.classifier_head_num = classifier_head_num
        self.classifier_learning_rate = classifier_learning_rate
        self.classifier_diversity_weight = classifier_diversity_weight
        self.classifier_l2_reg_weight = classifier_l2_reg_weight
        self.classifier_model_name = classifier_model_name
        
        
        set_seed(seed)
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        
        self.classifier = DivDisClassifier(use_gpu=use_gpu,
                                           log_dir=self.log_dir,
                                           num_classes=classifier_num_classes, 
                                           head_num=classifier_head_num,
                                           learning_rate=classifier_learning_rate,
                                           diversity_weight=classifier_diversity_weight,
                                           l2_reg_weight=classifier_l2_reg_weight,
                                           model_name=classifier_model_name
                                           )
        self.classifier.state_dim = 3 #(n,4,84,84) so each has dim3

        self.directory = None
        self.data_filenames = []
        
        self.positive_test_files = []
        self.negative_test_files = []
        self.uncertain_test_files = []
        
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        log_file = os.path.join(self.log_dir,
                                "{}.log".format(datetime.datetime.now()))
        
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.experiment_name, self.seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Seed: {}".format(seed))
        logging.info("Head num: {}".format(classifier_head_num))
        logging.info("Learning rate: {}".format(classifier_learning_rate))
        logging.info("Diversity weight: {}".format(classifier_diversity_weight))
        logging.info("L2 reg weight: {}".format(classifier_l2_reg_weight))
        logging.info("Initial epochs: {}".format(classifier_initial_epochs))
        logging.info("Per room epochs: {}".format(classifier_per_room_epochs))
    
    def save(self):
        self.classifier.save(path=self.save_dir)
    
    def load(self):
        self.classifier.load(path=self.save_dir)

    def read_directory(self, directory):
        self.data_filenames = get_sorted_filenames(directory)
        self.directory = directory

    def get_monte_filenames(self, task, room, init_term_uncertain, pos_neg=None):
        """
        Filters filenames by task, room, status (initiation/termination/uncertain), and label (positive/negative).

        Parameters:
        - filenames (list of str): Available filenames in resources folder
        - task (str): Task name (e.g., "climb_down_ladder").
        - room (str): Room number (e.g., "2", "14left").
        - init_term_uncertain (str): 'initiation', 'termination', or 'uncertain'.
        - pos_neg (str, optional): 'positive' or 'negative', ignored if 'uncertain'.

        Returns:
        - list of str: Filenames matching the criteria or an empty list.
        """

        if init_term_uncertain == 'uncertain':
            # If uncertain, we don't care about positive/negative
            pattern = re.compile(rf"{task}_room{room}(_1)?_{init_term_uncertain}\.npy")
        else:
            # For initiation/termination, match both positive/negative if pos_neg is not None
            pattern = re.compile(rf"{task}_room{room}(_1)?_{init_term_uncertain}_{pos_neg}\.npy")
        
        return [self.directory+filename for filename in self.data_filenames if pattern.match(filename)]
    
    def add_train_files(self,
                      positive_files,
                      negative_files,
                      unlabelled_files):
        
        self.classifier.add_data(positive_files=positive_files,
                                 negative_files=negative_files,
                                 unlabelled_files=unlabelled_files)

    def add_test_files(self,
                        positive_files,
                        negative_files,
                        uncertain_files=None):
        self.positive_test_files = positive_files
        self.negative_test_files = negative_files
        self.uncertain_test_files = uncertain_files
    
    def train_classifier(self, epochs=None):
        if epochs is None:
            epochs = self.initial_epochs
        self.classifier.train(epochs)
    
    def test_classifier(self):
        dataset_positive = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        dataset_negative = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        #dataset_positive.set_transform_function(transform)
        #dataset_negative.set_transform_function(transform)
        
        dataset_positive.add_true_files(self.positive_test_files)
        dataset_negative.add_false_files(self.negative_test_files)
    
        counter = 0
        accuracy = np.zeros(self.classifier.head_num)
        accuracy_pos = np.zeros(self.classifier.head_num)
        accuracy_neg = np.zeros(self.classifier.head_num)
        
        for _ in range(dataset_positive.num_batches):
            counter += 1
            x, y = dataset_positive.get_batch()
            pred_y, votes = self.classifier.predict(x)
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach().cpu()
                accuracy_pos[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)

        accuracy_pos /= counter
        
        total_count = counter
        counter = 0
        
        for _ in range(dataset_negative.num_batches):
            counter += 1
            x, y = dataset_negative.get_batch()
            pred_y, votes = self.classifier.predict(x)
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach().cpu()
                accuracy_neg[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)

        
        accuracy_neg /= counter
        total_count += counter
        
        accuracy /= total_count
        
        weighted_acc = (accuracy_pos + accuracy_neg)/2
        
        logging.info("============= Classifiers evaluated =============")
        for idx in range(self.classifier.head_num):
            logging.info("Head idx:{:<4}, True accuracy: {:.2f}, False accuracy: {:.2f}, Total accuracy: {:.2f}, Weighted accuracy: {:.2f}".format(
                idx,
                accuracy_pos[idx],
                accuracy_neg[idx],
                accuracy[idx],
                weighted_acc[idx])
            )
        logging.info("=================================================")
        
        return accuracy_pos, accuracy_neg, accuracy, weighted_acc


    def test_uncertainty(self):
        dataset_positive = SetDataset(max_size=1e6,
                                      batchsize=64)
        #dataset_positive.set_transform_function(transform)
        
        dataset_positive.add_true_files(self.uncertain_test_files)
    
        counter = 0
        uncertainty = np.zeros(self.classifier.head_num)
        
        for _ in range(dataset_positive.num_batches):
            counter += 1
            x, y = dataset_positive.get_batch()
            pred_y, votes = self.classifier.predict(x)
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach().cpu()
                uncertainty[idx] += (torch.sum(pred_class==y).item())/len(y)

        uncertainty /= counter
        
        logging.info("============= Classifiers Uncertainty States Measured =============")
        for idx in range(self.classifier.head_num):
            logging.info("Head idx:{:<4}, Uncertainty: {:.2f}.".format(idx, uncertainty[idx]))
        logging.info("=================================================")
        
        return uncertainty


    def view_false_predictions(self,
                        test_positive_files,
                        test_negative_files,
                        num_batch=1):
        dataset_positive = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        dataset_negative = SetDataset(max_size=1e6,
                                      batchsize=64)
        
        #dataset_positive.set_transform_function(transform)
        #dataset_negative.set_transform_function(transform)
        
        dataset_positive.add_true_files(test_positive_files)
        dataset_negative.add_false_files(test_negative_files)
    
        counter = 0
        accuracy = np.zeros(self.classifier.head_num)
        accuracy_pos = np.zeros(self.classifier.head_num)
        accuracy_neg = np.zeros(self.classifier.head_num)
        
        for _ in range(num_batch):
            counter += 1
            x, y = dataset_positive.get_batch()
            pred_y, votes = self.classifier.predict(x)
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach().cpu()
                accuracy_pos[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)

                # save false positive images to file'
                save_dir = os.path.join(self.plot_dir, 'false_negative', 'head_{}'.format(idx))
                os.makedirs(save_dir, exist_ok=True)
                
                false_neg_idx = (pred_class != y)
                false_neg_imgs = x[false_neg_idx].squeeze()
                
                with Pool(initializer=worker_initializer) as pool:
                    args = [(img, save_dir, counter, i + 1) for i, img in enumerate(false_neg_imgs)]
                    pool.starmap(save_image, args)


        accuracy_pos /= counter
        
        total_count = counter
        counter = 0
        
        for _ in range(num_batch):
            counter += 1
            x, y = dataset_negative.get_batch()
            pred_y, votes = self.classifier.predict(x)
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach().cpu()
                accuracy_neg[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)

                save_dir = os.path.join(self.plot_dir, 'false_positive', 'head_{}'.format(idx))
                os.makedirs(save_dir, exist_ok=True)
                
                false_pos_idx = (pred_class != y)
                false_pos_imgs = x[false_pos_idx]
                with Pool(initializer=worker_initializer) as pool:
                    args = [(img, save_dir, counter, i + 1) for i, img in enumerate(false_pos_imgs)]
                    pool.starmap(save_image, args)
                    
        
        accuracy_neg /= counter
        total_count += counter
        
        accuracy /= total_count
        
        weighted_acc = (accuracy_pos + accuracy_neg)/2
        
        logging.info("============= Classifiers evaluated =============")
        logging.info(f"Evaluated {num_batch} batches.")
        for idx in range(self.classifier.head_num):
            logging.info("Head idx:{:<4}, True accuracy: {:.2f}, False accuracy: {:.2f}, Total accuracy: {:.2f}, Weighted accuracy: {:.2f}".format(
                idx,
                accuracy_pos[idx],
                accuracy_neg[idx],
                accuracy[idx],
                weighted_acc[idx])
            )
        logging.info("=================================================")
        
        return accuracy_pos, accuracy_neg, accuracy, weighted_acc

    
    def explain_classifiers(self,
                            test_data,
                            test_head):
        dataset = SetDataset(max_size=1e6,
                             batchsize=64)
                
        dataset.add_true_files(test_data)
        
        true_data = []
        false_data = []
        
        for _ in range(dataset.num_batches):
            x, _ = dataset.get_batch()
            pred_y = self.classifier.predict(x)
            pred_y = pred_y.cpu()
            
            pred_class = torch.argmax(pred_y[:,test_head,:], dim=1).detach()
            
            true_data += x[pred_class == 1]
            false_data += x[pred_class == 0]
        
        if len(true_data) != 0:
            true_data = torch.stack(true_data)
        else:
            true_data = torch.zeros(1,1)
        if len(false_data) != 0:
            false_data = torch.stack(false_data)
        else:
            false_data = torch.zeros(1,1)
        
        return torch.std_mean(true_data, dim=0), torch.std_mean(false_data, dim=0)
        
        
    def plot_metrics(self, history, x_label, plot_name):
        if isinstance(history, dict):
            epochs = range(1, len(history['weighted_accuracy']) + 1)
            
            plt.figure(figsize=(12, 8))
            
            plt.plot(epochs, history['weighted_accuracy'], 'b-', label='Weighted Accuracy', linewidth=2)
            plt.plot(epochs, history['raw_accuracy'], 'g-', label='Raw Accuracy')
            #plt.plot(epochs, history['true_accuracy'], 'r-', label='True Accuracy')
            #plt.plot(epochs, history['false_accuracy'], 'c-', label='False Accuracy')
            if 'uncertainty' in history:
                plt.plot(epochs, history['uncertainty'], 'm-', label='Uncertainty Rate')
            
            plt.xlabel(x_label)
            plt.xticks(epochs)
            plt.ylabel('Metrics')
            plt.title('Training Metrics Over Time')
            plt.legend()
            
            plt.grid(True)
            
            plt.show(block=False)
            plt.savefig(os.path.join(self.plot_dir, plot_name))

        elif isinstance(history, list):
            # plot the average of given histories and use fill_between to show the variance
            epochs = range(1, len(history[0]['weighted_accuracy']) + 1)

            plt.figure(figsize=(12, 8))

            plt.plot(epochs, np.mean([hist['weighted_accuracy'] for hist in history], axis=0), 'b-', label='Weighted Accuracy', linewidth=3)
            plt.fill_between(epochs,
                             np.mean([hist['weighted_accuracy'] for hist in history], axis=0) - np.std([hist['weighted_accuracy'] for hist in history], axis=0),
                             np.mean([hist['weighted_accuracy'] for hist in history], axis=0) + np.std([hist['weighted_accuracy'] for hist in history], axis=0),
                             alpha=0.2, color='b')
            plt.plot(epochs, np.mean([hist['raw_accuracy'] for hist in history], axis=0), 'g-', label='Raw Accuracy')
            plt.fill_between(epochs,
                             np.mean([hist['raw_accuracy'] for hist in history], axis=0) - np.std([hist['raw_accuracy'] for hist in history], axis=0),
                             np.mean([hist['raw_accuracy'] for hist in history], axis=0) + np.std([hist['raw_accuracy'] for hist in history], axis=0),
                             alpha=0.2, color='g')
            if 'uncertainty' in history[0]:
                plt.plot(epochs, np.mean([hist['uncertainty'] for hist in history], axis=0), 'm-', label='Uncertainty Rate')
                plt.fill_between(epochs,
                                np.mean([hist['uncertainty'] for hist in history], axis=0) - np.std([hist['uncertainty'] for hist in history], axis=0),
                                np.mean([hist['uncertainty'] for hist in history], axis=0) + np.std([hist['uncertainty'] for hist in history], axis=0),
                                alpha=0.2, color='m')
            plt.xlabel(x_label)
            plt.xticks(epochs)
            plt.ylabel('Metrics')
            plt.title('Training Metrics Over Time')
            plt.legend()

            plt.grid(True)
            plt.show(block=False)
            plt.savefig(os.path.join(self.plot_dir, plot_name))
            


    def plot_heads(self, history, x_label, plot_name):
            num_rooms = len(history['weighted_accuracy'])
            num_heads = self.classifier.head_num

            epochs = range(1, num_rooms + 1)
            
            fig, axes = plt.subplots(1, 3 if 'uncertainty' in history else 2, figsize=(18, 6))

            colors = plt.cm.tab10(np.linspace(0, 1, num_heads))
            # going through rooms and determine width
            # first to reach 0.7 weighted acc, and the one with highest final weighted acc
            widths = np.ones(num_heads)
            for room_idx in range(num_rooms):
                for head_idx in range(num_heads):
                    weighted_acc = history['weighted_accuracy'][room_idx]
                    if np.max(weighted_acc) >= 0.7:
                        widths[np.argmax(weighted_acc)] = 3
            widths[np.argmax(history['weighted_accuracy'][-1])] = 3
            line_style = '-' 

            # Plot weighted accuracy for each head
            for head_idx in range(num_heads):
                weighted_acc = np.array([history['weighted_accuracy'][room_idx][head_idx] for room_idx in range(num_rooms)])
                color = colors[head_idx]
                axes[0].plot(epochs, weighted_acc, line_style, color=color, label=f'Head {head_idx}', alpha=0.7, linewidth=widths[head_idx])

            axes[0].set_xticks(epochs)
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel('Weighted Accuracy')
            axes[0].set_title('Weighted Accuracy Over Time')
            axes[0].legend(loc='lower right')
            axes[0].grid(True)

            # Plot raw accuracy for each head, bold using the same best head as in weighted accuracy
            for head_idx in range(num_heads):
                raw_acc = np.array([history['raw_accuracy'][room_idx][head_idx] for room_idx in range(num_rooms)])
                color = colors[head_idx]
                axes[1].plot(epochs, raw_acc, line_style, color=color, label=f'Head {head_idx}', alpha=0.7, linewidth=widths[head_idx])

            axes[1].set_xticks(epochs)
            axes[1].set_xlabel(x_label)
            axes[1].set_ylabel('Raw Accuracy')
            axes[1].set_title('Raw Accuracy Over Time')
            axes[1].legend(loc='lower right')
            axes[1].grid(True)

            # Plot uncertainty if present, bold using the same best head as in weighted accuracy
            if 'uncertainty' in history:
                for head_idx in range(num_heads):
                    uncertainty = np.array([history['uncertainty'][room_idx][head_idx] for room_idx in range(num_rooms)])
                    color = colors[head_idx]
                    axes[2].plot(epochs, uncertainty, line_style, color=color, label=f'Head {head_idx}', alpha=0.7, linewidth=widths[head_idx])

                axes[2].set_xticks(epochs)
                axes[2].set_xlabel(x_label)
                axes[2].set_ylabel('Uncertainty')
                axes[2].set_title('Uncertainty Over Time')
                axes[2].legend(loc='lower right')
                axes[2].grid(True)

            plt.tight_layout()
            plt.show(block=False)
            plt.savefig(os.path.join(self.plot_dir, plot_name))

            
    def room_by_room_train_unlabelled(self, room_list, unlabelled_train_files, history, heads_history):
        for room_idx in tqdm(range(len(room_list)), desc='Room Progression'):
            room = room_list[room_idx]
            print('===============================')
            print('===============================')
            print(f"Training on room {room}")
            logging.info(f"Training on room {room}")
            
            cur_room_unlab = unlabelled_train_files[room_idx]
            #cur_room_unlab = [np.load(file) for file in cur_room_unlab]
            #cur_room_unlab = [img for list in cur_room_unlab for img in list]
            #cur_room_unlab = [torch.from_numpy(img).float().squeeze() for img in cur_room_unlab]
            #self.classifier.dataset.add_unlabelled_data(cur_room_unlab)
            self.classifier.add_data([], [], cur_room_unlab)
            
            self.train_classifier(self.per_room_epochs)
                
            accuracy_pos, accuracy_neg, accuracy, weighted_acc = self.test_classifier()
                                                        
            print(f"Weighted Accuracy: \n{np.round(weighted_acc, 2)}")
            print(f"Accuracy: \n{np.round(accuracy, 2)}")
            
            heads_history['weighted_accuracy'].append(weighted_acc)
            heads_history['raw_accuracy'].append(accuracy)

            best_weighted_acc = np.max(weighted_acc)
            best_head_idx = np.argmax(weighted_acc)
            best_accuracy = accuracy[best_head_idx]
            best_true_acc = accuracy_pos[best_head_idx]
            best_false_acc = accuracy_neg[best_head_idx]
            
            history['weighted_accuracy'].append(best_weighted_acc)
            history['raw_accuracy'].append(best_accuracy)
            history['true_accuracy'].append(best_true_acc)
            history['false_accuracy'].append(best_false_acc)
            
            if self.uncertain_test_files is not None:
                uncertainty = self.test_uncertainty()
                print(f"Uncertainty: \n{np.round(uncertainty, 2)}")
                best_head_uncertainty = uncertainty[best_head_idx]
                history['uncertainty'].append(best_head_uncertainty)
                heads_history['uncertainty'].append(uncertainty)

        # save history to pickle
        with open(os.path.join(self.log_dir, 'room_progression_metrics.dict'), 'wb') as f:
            pickle.dump(history, f)
        self.plot_metrics(history, 'room', 'room_progression_metrics')

        with open(os.path.join(self.log_dir, 'heads_progression_metrics.dict'), 'wb') as f:
            pickle.dump(heads_history, f)
        self.plot_heads(heads_history, 'room', 'heads_metrics')

        print("All unlabelled rooms added, now running additional training loops")
        logging.info("All unlabelled rooms added, now running additional training loops")

        return history, heads_history
        

    def additional_train(self, num_loops=20): 
        history = {
            'weighted_accuracy': [],
            'raw_accuracy': [],
            'true_accuracy': [],
            'false_accuracy': [],
        }
        heads_history = {
            'weighted_accuracy': [],
            'raw_accuracy': [],
        }
        if self.uncertain_test_files is not None:
            history['uncertainty'] = []
            heads_history['uncertainty'] = []

            
        for i in tqdm(range(num_loops), desc='Additional Training Loops'):

            print(f"Additional Training Loop {i}")
            logging.info(f"Additional Training Loop {i}")
            
            self.train_classifier(self.per_room_epochs)

            accuracy_pos, accuracy_neg, accuracy, weighted_acc = self.test_classifier()
                                                        
            print(f"Weighted Accuracy: \n{np.round(weighted_acc, 2)}")
            print(f"Accuracy: \n{np.round(accuracy, 2)}")

            heads_history['weighted_accuracy'].append(weighted_acc)
            heads_history['raw_accuracy'].append(accuracy)

            best_weighted_acc = np.max(weighted_acc)
            best_head_idx = np.argmax(weighted_acc)
            best_accuracy = accuracy[best_head_idx]
            best_true_acc = accuracy_pos[best_head_idx]
            best_false_acc = accuracy_neg[best_head_idx]
            
            history['weighted_accuracy'].append(best_weighted_acc)
            history['raw_accuracy'].append(best_accuracy)
            history['true_accuracy'].append(best_true_acc)
            history['false_accuracy'].append(best_false_acc)

            if self.uncertain_test_files is not None:
                uncertainty = self.test_uncertainty()
                print(f"Uncertainty: \n{np.round(uncertainty, 2)}")
                best_head_uncertainty = uncertainty[best_head_idx]
                history['uncertainty'].append(best_head_uncertainty)
                if heads_history is not None:
                    heads_history['uncertainty'].append(uncertainty)


        # save history to pickle
        with open(os.path.join(self.log_dir, 'additional_loops.dict'), 'wb') as f:
            pickle.dump(history, f)
        self.plot_metrics(history, 'additional_loops', 'additional_train_metrics')

        with open(os.path.join(self.log_dir, 'heads_additional_loops.dict'), 'wb') as f:
            pickle.dump(heads_history, f)
        self.plot_heads(heads_history, 'additional_loops', 'heads_additional_train_metrics')
        
        return history, heads_history


    def room_by_room_train_labelled(self, task, init_term, combinations_list=None, files_list=None, num_seeds=5):
        
        # combinations_list: 2D array, (num_rooms, num_combinations)

        if combinations_list is not None:
        
            num_instances = range(1,len(combinations_list)+1)
            weighted_acc_list = [] # nested list of best heads
            raw_acc_list = []
            
            for room_combinations in combinations_list:
                # each combination include all rooms (with n rooms). e.g. [[1]] for 1 room, [[1,2],[0,1]] for 2 rooms, etc.
                comb_weighted_acc = []
                comb_raw_acc = []
                num_rooms = len(room_combinations[0])
                
                print(f"==============\nTraining on combinations of {num_rooms} instances of labeled ladder data")
                logging.info(f"=============\nTraining on combinations of {num_rooms} instances of labeled ladder data")
                
                for comb in room_combinations:
                    # each combination has n rooms, want to add all these to labelled and measure test performance.
                    positive_train_files = []
                    negative_train_files = []

                    print(f"***\nTraining on ladders {comb}")
                    logging.info(f"***\nTraining on ladders {comb}")
                    
                    for room in comb:
                        positive_train_files += self.get_monte_filenames(task, room, init_term, 'positive')
                        negative_train_files += self.get_monte_filenames(task, room, init_term, 'negative')
                        if room == 1:
                            img_dir = self.directory
                            negative_train_files += [img_dir+"screen_death_1.npy",
                                                    img_dir+"screen_death_2.npy",
                                                    img_dir+"screen_death_3.npy",
                                                    img_dir+"screen_death_4.npy",]
                    unlabelled_train_files = unlabelled_train_files = [self.directory+file for file in self.data_filenames if (file not in positive_train_files) and (file not in negative_train_files)]
                    unlabelled_train_files = random.sample(unlabelled_train_files, int(1*len(unlabelled_train_files)))

                    print(f"Positive files: {positive_train_files}")
                    print(f"Negative files: {negative_train_files}")
                    
                    
                    for s in range(num_seeds):
                        set_seed(self.seed+s)
                        print(f"Seed {self.seed+s}")
                        logging.info(f"Seed {self.seed+s}")
                        
                        #classifier = copy.deepcopy(self.classifier)
                        classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                                    log_dir=self.log_dir,
                                                    num_classes=self.classifier.num_classes,
                                                    head_num=self.classifier.head_num,
                                                    learning_rate=self.classifier_learning_rate,
                                                    diversity_weight=self.classifier_diversity_weight,
                                                    l2_reg_weight=self.classifier_l2_reg_weight,
                                                    model_name=self.classifier_model_name
                                                    )
                        self.classifier = classifier
                        classifier.add_data(positive_train_files, negative_train_files, unlabelled_train_files)
                        classifier.set_class_weights()
                        classifier.train(self.per_room_epochs)

                        accuracy_pos, accuracy_neg, raw_acc, weighted_acc = self.test_classifier()
                        

                        best_head_idx = np.argmax(weighted_acc)
                        best_weighted_acc = np.max(weighted_acc)
                        best_accuracy = raw_acc[best_head_idx]
                        best_true_acc = accuracy_pos[best_head_idx]
                        best_false_acc = accuracy_neg[best_head_idx]
                        
                        comb_weighted_acc.append(best_weighted_acc)
                        comb_raw_acc.append(best_accuracy)

                        print(f"Weighted Accuracy: {np.round(weighted_acc, 2)}")
                        print(f"Accuracy:          {np.round(raw_acc, 2)}")
                        print(f"True Accuracy:     {np.round(accuracy_pos, 2)}")
                        print(f"False Accuracy:    {np.round(accuracy_neg, 2)}")

                        logging.info(f"Weighted Accuracy: {np.round(weighted_acc, 2)}")
                        logging.info(f"Accuracy:          {np.round(raw_acc, 2)}")
                        logging.info(f"True Accuracy:     {np.round(accuracy_pos, 2)}")
                        logging.info(f"False Accuracy:    {np.round(accuracy_neg, 2)}")

                        if self.uncertain_test_files is not None:
                            uncertainty = self.test_uncertainty()
                            print(f"Uncertainty:       {np.round(uncertainty, 2)}")
                            logging.info(f"Uncertainty:       {np.round(uncertainty, 2)}")

                weighted_acc_list.append(comb_weighted_acc)
                raw_acc_list.append(comb_raw_acc)


        else:
            assert files_list is not None
            positive_list, negative_list, unlabelled_list = files_list
        
            num_instances = range(1,len(positive_list)+1)
            results_weighted_acc = np.zeros((len(positive_list), 3, num_seeds)) # (num_instances, 3 classifiers, 5 seeds)
            results_raw_acc = np.zeros((len(positive_list), 3, num_seeds)) 

            positive_train_files = []
            negative_train_files = []
            unlabelled_train_files = []
            
            for i in range(len(positive_list)):
                print("=====================================")
                logging.info("=====================================")
                print(f"Training on {num_instances[i]} instances of labeled data")
                logging.info(f"Training on {num_instances[i]} instances of labeled data")
                
                comb_weighted_acc = [[] for _ in range(3)] # (classifier, seed)
                comb_raw_acc = [[] for _ in range(3)]

                positive_train_files += positive_list[i]
                negative_train_files += negative_list[i]
                unlabelled_train_files += unlabelled_list[i]
                
                
                print(f"Positive files: {positive_train_files}")
                print(f"Negative files: {negative_train_files}")
                
                
                for c in range(3):
                    classifier_names = ['D-BAT Ensemble','D-BAT Ensemble - no diversity','CNN']
                    print(f"Training {classifier_names[c]}")
                    logging.info(f"Training {classifier_names[c]}")
                    
                    for s in range(num_seeds):
                        set_seed(self.seed+s)
                        print(f"Seed {self.seed+s}")
                        logging.info(f"Seed {self.seed+s}")

                        classifier_div_weight = 0 if c == 1 else self.classifier_diversity_weight
                        classifier_head_num = 1 if c == 2 else self.classifier_head_num
                    
                        classifier = DivDisClassifier(use_gpu=self.use_gpu,
                                                    log_dir=self.log_dir,
                                                    num_classes=self.classifier.num_classes,
                                                    head_num=classifier_head_num,
                                                    learning_rate=self.classifier_learning_rate,
                                                    diversity_weight=classifier_div_weight,
                                                    l2_reg_weight=self.classifier_l2_reg_weight,
                                                    model_name=self.classifier_model_name
                                                    )
                        self.classifier = classifier
                        classifier.add_data(positive_train_files, negative_train_files, unlabelled_train_files)
                        classifier.set_class_weights()
                        classifier.train(self.per_room_epochs, progress_bar=True)

                        accuracy_pos, accuracy_neg, raw_acc, weighted_acc = self.test_classifier()

                        best_head_idx = np.argmax(weighted_acc)
                        best_weighted_acc = np.max(weighted_acc)
                        best_accuracy = raw_acc[best_head_idx]
                        best_true_acc = accuracy_pos[best_head_idx]
                        best_false_acc = accuracy_neg[best_head_idx]
                        
                        results_weighted_acc[i,c,s] = best_weighted_acc
                        results_raw_acc[i,c,s] = best_accuracy

                        print(f"Weighted Accuracy: {np.round(weighted_acc, 2)}")
                        print(f"Accuracy:          {np.round(raw_acc, 2)}")
                        print(f"True Accuracy:     {np.round(accuracy_pos, 2)}")
                        print(f"False Accuracy:    {np.round(accuracy_neg, 2)}")

                        logging.info(f"Weighted Accuracy: {np.round(weighted_acc, 2)}")
                        logging.info(f"Accuracy:          {np.round(raw_acc, 2)}")
                        logging.info(f"True Accuracy:     {np.round(accuracy_pos, 2)}")
                        logging.info(f"False Accuracy:    {np.round(accuracy_neg, 2)}")

                        if self.uncertain_test_files is not None:
                            uncertainty = self.test_uncertainty()
                            print(f"Uncertainty:       {np.round(uncertainty, 2)}")
                            logging.info(f"Uncertainty:       {np.round(uncertainty, 2)}")



        results = {'instances': num_instances, 'weighted_acc': results_weighted_acc, 'raw_acc': results_raw_acc}

        with open(os.path.join(self.log_dir, f'{task}_room_train.dict'), 'wb') as f:
            pickle.dump(results, f)

        self.plot(f"{task}_labeled_vs_acc", num_instances, results_weighted_acc, f"Weighted Accuracy vs Instances of Labeled Data ({task.capitalize()})", "Instances of Labeled Data")
        

    def plot(self,
            plot_file,
            x_values,
            y_values,         
            plot_title,
            x_label):

        font_params = { # larger fonts for better readability
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'axes.titleweight': 'bold',
            'legend.frameon': True,      
            'legend.loc': 'lower right',
            'grid.alpha': 0.6,           
            'grid.linestyle': '--'      
        }
        plt.rcParams.update(font_params)

        
        # Convert x_values to a NumPy array for consistency
        x_values = np.array(x_values)
        
        # Extract each y-value array
        divdis_ensemble_acc = np.array(y_values[0]) # (num_classifiers, num_seeds)
        regular_ensemble_acc = np.array(y_values[1])
        cnn_acc = np.array(y_values[2])
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(12, 7))  
        
        # Calculate means and standard deviations across seeds
        divdis_mean = np.mean(divdis_ensemble_acc, axis=1)
        divdis_std = np.std(divdis_ensemble_acc, axis=1)
        
        regular_mean = np.mean(regular_ensemble_acc, axis=1)
        regular_std = np.std(regular_ensemble_acc, axis=1)
        
        cnn_mean = np.mean(cnn_acc, axis=1)
        cnn_std = np.std(cnn_acc, axis=1)

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(12, 7))  # Bigger figure size

        # Plot mean values with shaded standard deviation
        ax.plot(x_values, divdis_mean, label='D-BAT Ensemble', color='b', marker='o', linestyle='-', linewidth=2)
        ax.fill_between(x_values, divdis_mean - divdis_std, divdis_mean + divdis_std, color='b', alpha=0.2)
        
        ax.plot(x_values, regular_mean, label='D-BAT Ensemble - no diversity', color='g', marker='x', linestyle='--', linewidth=2)
        ax.fill_between(x_values, regular_mean - regular_std, regular_mean + regular_std, color='g', alpha=0.2)
        
        ax.plot(x_values, cnn_mean, label='CNN', color='r', marker='s', linestyle='-.', linewidth=2)
        ax.fill_between(x_values, cnn_mean - cnn_std, cnn_mean + cnn_std, color='r', alpha=0.2)
        
        
        # Improve the overall aesthetics
        ax.set_xlabel(x_label)
        ax.set_ylabel('Accuracy')
        ax.set_title(plot_title)

        # Adjust y-axis limits if needed 
        ax.set_ylim(0.65, 1)

        # Adding grid and improving the axis labels
        ax.grid(True)
        ax.tick_params(axis='both', which='major')

        # Adding legend
        ax.legend()
        
        # Save and show the figure
        fig.tight_layout()
        fig.savefig(plot_file, dpi=300)
        print(f"Figure saved at: {os.path.abspath(plot_file)}")  # Print the path of the saved figure
        
        plt.show()
        plt.close(fig)
