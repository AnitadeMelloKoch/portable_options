import logging 
import numpy as np 
from scipy import stats
import torch 
import os
from portable.option.memory import SetDataset
from portable.option.ensemble.custom_attention import *
from portable.option.sets.utils import BayesianWeighting

logger = logging.getLogger(__name__)

class AttentionSet():
    def __init__(self,
                 use_gpu,
                 vote_threshold,
                 embedding: AutoEncoder,
                 log_dir,
                 
                 attention_module_num=8,
                 learning_rate=1e-3,
                 beta_distribution_alpha=30,
                 beta_distribution_beta=5,
                 divergence_loss_scale=0.05,
                 regularization_loss_scale=0.,
                 
                 dataset_max_size=200000,
                 dataset_batch_size=16,
                 
                 summary_writer=None,
                 model_name="classifier",
                 padding_func=None):
        
        self.embedding = embedding
        self.embedding.eval()
        
        self.use_gpu = use_gpu
        self.vote_threshold = vote_threshold
        if padding_func is None:
            padding_func = lambda x: x
        self.dataset = SetDataset(max_size=dataset_max_size, 
                                  batchsize=dataset_batch_size,
                                  pad_func=padding_func)
        self.attention_num = attention_module_num
        self.alpha = beta_distribution_alpha
        self.beta = beta_distribution_beta
        self.div_scale = divergence_loss_scale
        self.reg_scale = regularization_loss_scale
        self.log_dir = log_dir
        self.summary_writer = summary_writer
        self.name = model_name
        
        self.classifier = AttentionEnsembleII(num_attention_heads=attention_module_num,
                                              num_classes=2,
                                              embedding_size=embedding.feature_size)
        self.confidences = BayesianWeighting(beta_distribution_alpha,
                                             beta_distribution_beta,
                                             self.attention_num)
        
        self.votes = None 
        
        self.optimizers = [
            torch.optim.Adam(self.classifier.attentions[idx].parameters(), lr=learning_rate) for idx in range(self.attention_num)
        ]
        
        self.crossentropy = torch.nn.CrossEntropyLoss()

        # Saved feature mean and sd during initial training
        self.saved_ftr_mean = None
        self.saved_ftr_sd = None
        self.saved_ftr_n = 0
        self.this_ftr_mean = None
        self.this_ftr_sd = None
        self.this_ftr_n = 0

        
    def save(self, path):
        torch.save(self.classifier.state_dict(), os.path.join(path, 'classifier_ensemble.ckpt'))
        self.dataset.save(path)
        self.confidences.save(os.path.join(path, 'confidence'))
    
    def load(self, path):
        if os.path.exists(os.path.join(path, 'classifier_ensemble.ckpt')):
            print("Classifier loaded from: {}".format(path))
            self.classifier.load_state_dict(torch.load(os.path.join(path, 'classifier_ensemble.ckpt')))
            self.dataset.load(path)
            self.confidences.load(os.path.join(path, 'confidence'))
    
    def move_to_gpu(self):
        if self.use_gpu:
            self.classifier.to("cuda")
    
    def move_to_cpu(self):
        self.classifier.to("cpu")
    
    def add_data(self,
                 positive_data=[],
                 negative_data=[],
                 priority_negative_data=[]):
        assert isinstance(positive_data, list)
        assert isinstance(negative_data, list)
        assert isinstance(priority_negative_data, list)

        if len(positive_data) > 0:
            self.dataset.add_true_data(positive_data, self.sample_confidence(positive_data))
        
        if len(negative_data) > 0:
            self.dataset.add_false_data(negative_data, self.sample_confidence(negative_data))

        if len(priority_negative_data) > 0:
            self.dataset.add_priority_false_data(priority_negative_data, 
                                                 self.sample_confidence(priority_negative_data))

    def add_data_from_files(self,
                            positive_files,
                            negative_files,
                            priority_negative_files=[]):
        assert isinstance(positive_files, list)
        assert isinstance(negative_files, list)
        assert isinstance(priority_negative_files, list)

        self.dataset.add_true_files(positive_files)
        self.dataset.add_false_files(negative_files)
        self.dataset.add_priority_false_files(priority_negative_files)
    
    def train(self,
              epochs,
              save_ftr_distribution=False):

        self.move_to_gpu()
        self.classifier.train()

        # Calculate running Mean and sd for features
        temp_ftr_mean = np.zeros((self.embedding.feature_size))
        temp_ftr_SST = np.zeros((self.embedding.feature_size))
        running_n = 0
        
        for epoch in range(epochs):
            self.dataset.shuffle()
            loss = np.zeros(self.attention_num)
            classifier_losses = np.zeros(self.attention_num)
            classifier_acc = np.zeros(self.attention_num)
            div_losses = np.zeros(self.attention_num)
            l1_losses = np.zeros(self.attention_num)
            counter = 0
            for _ in range(self.dataset.num_batches):
                counter += 1
                x, y = self.dataset.get_batch()
                if self.use_gpu:
                    x = x.to("cuda")
                    y = y.to("cuda")
                x = self.embedding.feature_extractor(x)
                pred_y = self.classifier(x)
                masks = self.classifier.get_attention_masks()

                # Compute features post mask 
                # convert to numpy array? is it torch tensor type?
                x_post_mask = masks*x

                for batch_x_idx in range(x_post_mask.shape[0]):
                    running_n += 1
                    this_row = x_post_mask[batch_x_idx]
                    if running_n == 1:
                        temp_ftr_mean = this_row
                    else:
                        prev_mean = temp_ftr_mean
                        temp_ftr_mean = prev_mean + (this_row-prev_mean)/(running_n)
                        temp_ftr_SST += (this_row-prev_mean)*(this_row-temp_ftr_mean)

                for attn_idx in range(self.attention_num):
                    b_loss = self.crossentropy(pred_y[attn_idx], y)
                    pred_class = torch.argmax(pred_y[attn_idx], dim=1).detach()
                    classifier_losses[attn_idx] += b_loss.item()
                    div_loss = self.div_scale*divergence_loss(masks, attn_idx)
                    div_losses[attn_idx] += div_loss.item()
                    regulariser_loss = self.reg_scale*l1_loss(masks, attn_idx)
                    l1_losses[attn_idx] += regulariser_loss
                    b_loss += div_loss
                    b_loss += regulariser_loss
                    classifier_acc[attn_idx] += (torch.sum(pred_class==y).item())/len(y)
                    b_loss.backward()
                    self.optimizers[attn_idx].step()
                    self.optimizers[attn_idx].zero_grad()
                    loss[attn_idx] += b_loss.item()
            
            if self.summary_writer is not None:
                print("Epoch {}".format(epoch))
                logger.info("Epoch {}".format(epoch))
                for idx in range(self.attention_num):
                    self.summary_writer.add_scalar('{}/total_loss/{}'.format(self.name, idx),
                                                   loss[idx]/counter,
                                                   epoch)
                    self.summary_writer.add_scalar('{}/classifier_loss/{}'.format(self.name, idx),
                                                   classifier_losses[idx]/counter,
                                                   epoch)
                    self.summary_writer.add_scalar('{}/divergence_loss/{}'.format(self.name, idx),
                                                   div_losses[idx]/counter,
                                                   epoch)
                    self.summary_writer.add_scalar('{}/l1_loss/{}'.format(self.name, idx),
                                                   l1_losses[idx]/counter,
                                                   epoch)
                    self.summary_writer.add_scalar('{}/accuracy/{}'.format(self.name, idx),
                                                   classifier_acc[idx]/counter,
                                                   epoch)
            
                    print("att {} - class loss: {:.2f} div loss: {:.2f} l1 loss: {:.2f} total loss: {:.2f} acc: {:.2f}".format(idx, 
                                                                            classifier_losses[idx]/counter,
                                                                            div_losses[idx]/counter,
                                                                            l1_losses[idx]/counter,
                                                                            loss[idx]/counter,
                                                                            classifier_acc[idx]/counter))

                    logger.info("att {} - class loss: {:.2f} div loss: {:.2f} l1 loss: {:.2f} total loss: {:.2f} acc: {:.2f}".format(idx, 
                                                                            classifier_losses[idx]/counter,
                                                                            div_losses[idx]/counter,
                                                                            l1_losses[idx]/counter,
                                                                            loss[idx]/counter,
                                                                            classifier_acc[idx]/counter))
        temp_ftr_sd = (temp_ftr_SST/(running_n-1))**0.5
        if save_ftr_distribution:
            self.saved_ftr_mean = temp_ftr_mean
            self.saved_ftr_sd = temp_ftr_sd
            self.saved_ftr_n = running_n
        self.this_ftr_mean = temp_ftr_mean
        self.this_ftr_sd = temp_ftr_sd
        self.this_ftr_n = running_n

    def vote(self, x):
        self.classifier.eval()
        
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).float()
        
        if len(x.shape):
            x = x.unsqueeze(0)
        if self.use_gpu:
            x = x.to("cuda")
        x = self.embedding.feature_extractor(x)
        with torch.no_grad():
            conf = self.confidences.weights()
            pred_y = self.classifier(x,  concat_results=True)
        
        votes = torch.argmax(pred_y, axis=-1)[0]
        
        self.votes = votes
        
        vote = False
        
        for idx in range(self.attention_num):
            if conf[idx] >= self.vote_threshold:
                if votes[idx] == 1:
                    vote = True
        
        return vote, pred_y, votes, conf
        
    def get_attentions(self):
        return self.classifier.get_attention_masks()

    def update_confidence(self,
                          was_successful: bool,
                          votes: list):
        success_count = votes.cpu().numpy()
        failure_count = np.ones(len(success_count)) - success_count
        
        if not was_successful:
            success_count = failure_count
            failure_count = votes.cpu().numpy()
        
        self.confidences.update_successes(success_count)
        self.confidences.update_failures(failure_count)

    def get_saved_ftr_distrbution(self):
        return self.saved_ftr_mean, self.saved_ftr_sd, self.saved_ftr_n

    def get_this_ftr_distribution(self):
        return self.this_ftr_mean, self.this_ftr_sd, self.this_ftr_n

    def sample_confidence(self, data=[]):
        # Assume given input is data_list
        if len(data) > 0:
            # assume each data is a 1-d numpy array length = feature_size
            data_matrix = np.vstack(data)
            given_mean = np.mean(data_matrix, axis=0)
            given_sd = np.std(data_matrix, axis=0)
            given_n = len(data)
        # if not given input, use saved feature distribution
        else:
            given_mean, given_sd, given_n = self.get_this_ftr_distribution()

        saved_mean, saved_sd, saved_n = self.get_saved_ftr_distrbution()
        # Calculate p-values for each feature
        SE_diff = np.sqrt(saved_sd**2/saved_n + given_sd**2/given_n)
        # low z-score indicates similar samples
        z_stat = (given_mean - saved_mean) / SE_diff
        # low z-score --> high p-value
        p_value = stats.norm.sf(np.abs(z_stat))*2  # Two-tailed

        # cutoff to get one number ?
        samp_conf_mean = np.mean(p_value) # all medium conf vs. some high some low? WANT high conf and low sd
        samp_conf_sd = np.std(p_value)
        bonus = 0
        if (samp_conf_mean) > 0.6 and (samp_conf_sd) < 0.2:
            bonus = 0.1
        return samp_conf_mean + bonus