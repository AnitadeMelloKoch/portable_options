import datetime
import logging
import os
import pickle

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

def transform(x):
    # only keep the last channel of imgs
    x = x[:, -1]
    x = x.unsqueeze(1)
    return x

@gin.configurable 
class MonteDivDisClassifierExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 use_gpu,
                 F
                 classifier_num_classes,
                 
                 classifier_head_num,
                 classifier_learning_rate,
                 classifier_diversity_weight,
                 classifier_l2_reg_weight,
                 classifier_initial_epochs,
                 classifier_per_room_epochs
                 ):
        
        self.seed = seed 
        self.base_dir = base_dir
        self.experiment_name = experiment_name
        self.initial_epochs = classifier_initial_epochs
        self.per_room_epochs = classifier_per_room_epochs
        
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
                                           head_num=classifier_head_num,
                                           learning_rate=classifier_learning_rate,
                                           num_classes=classifier_num_classes,
                                           diversity_weight=classifier_diversity_weight,
                                           l2_reg_weight=classifier_l2_reg_weight,
                                           model_name='monte_cnn'
                                           )
        #self.classifier.dataset.set_transform_function(transform)

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
            logging.info("Head idx:{:<4}, True accuracy: {:.4f}, False accuracy: {:.4f}, Total accuracy: {:.4f}, Weighted accuracy: {:.4f}".format(
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
            logging.info("Head idx:{:<4}, Uncertainty: {:.4f}.".format(idx, uncertainty[idx]))
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
            logging.info("Head idx:{:<4}, True accuracy: {:.4f}, False accuracy: {:.4f}, Total accuracy: {:.4f}, Weighted accuracy: {:.4f}".format(
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
            

            
    def room_by_room_train(self, room_list, unlabelled_train_files, history):
        for room_idx in tqdm(range(len(room_list)), desc='Room Progression'):
            room = room_list[room_idx]
            print('===============================')
            print('===============================')
            print(f"Training on room {room}")
            logging.info(f"Training on room {room}")
            
            cur_room_unlab = unlabelled_train_files[room_idx]
            cur_room_unlab = [np.load(file) for file in cur_room_unlab]
            cur_room_unlab = [img for list in cur_room_unlab for img in list]
            cur_room_unlab = [torch.from_numpy(img).float().squeeze() for img in cur_room_unlab]
            self.classifier.dataset.add_unlabelled_data(cur_room_unlab)
            
            self.train_classifier(self.per_room_epochs)
                
            accuracy_pos, accuracy_neg, accuracy, weighted_acc = self.test_classifier()
                                                        
            print(f"Weighted Accuracy: \n{np.round(weighted_acc, 2)}")
            print(f"Accuracy: \n{np.round(accuracy, 2)}")

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

        # save history to pickle
        with open(os.path.join(self.log_dir, 'room_progression_metrics'), 'wb') as f:
            pickle.dump(history, f)
        self.plot_metrics(history, 'room', 'room_progression_metrics')

        print("All unlabelled rooms added, now running additional training loops")
        logging.info("All unlabelled rooms added, now running additional training loops")

        return history
        

    def additional_train(self, num_loops=20): 
        history = {
            'weighted_accuracy': [],
            'raw_accuracy': [],
            'true_accuracy': [],
            'false_accuracy': [],
        }
        if self.uncertain_test_files is not None:
            history['uncertainty'] = []

            
        for i in tqdm(range(num_loops), desc='Additional Training Loops'):

            print(f"Additional Training Loop {i}")
            logging.info(f"Additional Training Loop {i}")
            
            self.train_classifier(self.per_room_epochs)

            accuracy_pos, accuracy_neg, accuracy, weighted_acc = self.test_classifier()
                                                        
            print(f"Weighted Accuracy: \n{np.round(weighted_acc, 2)}")
            print(f"Accuracy: \n{np.round(accuracy, 2)}")

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


        # save history to pickle
        with open(os.path.join(self.log_dir, 'additional_loops'), 'wb') as f:
            pickle.dump(history, f)
        self.plot_metrics(history, 'additional_loops', 'additional_train_metrics')

        return history
