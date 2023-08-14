import logging 
import numpy as np 
import torch 
import os
from portable.option.memory import SetDataset
from portable.option.ensemble.custom_attention import *
from portable.option.sets.utils import BayesianWeighting

logger = logging.getLogger(__name__)

class AttentionSet():
    def __init__(self,
                 device,
                 vote_function,
                 embedding: AutoEncoder,
                 log_dir,
                 
                 attention_module_num=8,
                 learning_rate=1e-3,
                 beta_distribution_alpha=30,
                 beta_distribution_beta=5,
                 divergence_loss_scale=1,
                 regularization_loss_scale=0.5,
                 
                 dataset_max_size=200000,
                 dataset_batch_size=16,
                 
                 summary_writer=None,
                 model_name="classifier"):
        
        self.embedding = embedding
        
        self.classifier = AttentionEnsembleII(num_attention_heads=attention_module_num,
                                              num_features=embedding.num_embedding_features,
                                              num_classes=2,
                                              input_dim=attention_module_num*embedding.feature_size)
        self.classifier.to(device)
        self.confidences = BayesianWeighting(beta_distribution_alpha,
                                             beta_distribution_beta,
                                             self.attention_num,
                                             self.device)
        
        self.device = device
        self.vote_function = vote_function
        self.dataset = SetDataset(max_size=dataset_max_size, batchsize=dataset_batch_size)
        self.attention_num = attention_module_num
        self.alpha = beta_distribution_alpha
        self.beta = beta_distribution_beta
        self.div_scale = divergence_loss_scale
        self.reg_scale = regularization_loss_scale
        self.log_dir = log_dir
        self.summary_writer = summary_writer
        self.name = model_name
        
        self.votes = None 
        
        self.optimizers = [
            torch.optim.Adam(self.classifier.attentions[idx].parameters(), lr=learning_rate) for idx in range(self.attention_num)
        ]
        
        self.crossentropy = torch.nn.CrossEntropyLoss()
        
    def save(self, path):
        torch.save(self.classifier.state_dict(), os.path.join(path, 'classifier_ensemble.ckpt'))
        self.dataset.save(path)
        self.confidences.save(os.path.join(path, 'confidence'))
    
    def load(self, path):
        self.classifier.load_state_dict(torch.load(os.path.join(path, 'classifier_ensemble.ckpt')))
        self.dataset.load(path)
        self.confidences.load(os.path.join(path, 'confidence'))
    
    def add_data(self,
                 positive_data=[],
                 negative_data=[],
                 priority_negative_data=[]):
        assert isinstance(positive_data, list)
        assert isinstance(negative_data, list)
        assert isinstance(priority_negative_data, list)

        if len(positive_data) > 0:
            self.dataset.add_true_data(positive_data)
        
        if len(negative_data) > 0:
            self.dataset.add_false_data(negative_data)

        if len(priority_negative_data) > 0:
            self.dataset.add_priority_false_data(priority_negative_data)

    def add_data_from_files(self,
                            positive_files,
                            negative_files,
                            priority_negative_files):
        assert isinstance(positive_files, list)
        assert isinstance(negative_files, list)
        assert isinstance(priority_negative_files, list)

        self.dataset.add_true_files(positive_files)
        self.dataset.add_false_files(negative_files)
        self.dataset.add_priority_false_files(priority_negative_files)
    
    def train(self,
              epochs):
        
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
                x = x.to(self.device)
                x = self.embedding.feature_extractor(x)
                y = y.to(self.device)
                pred_y = self.classifier(x)
                masks = self.classifier.get_attention_masks()
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
            
            masks = self.classifier.get_attention_masks()
            cat_masks = torch.cat(masks).squeeze().detach().cpu().numpy()
            fig, ax = plt.subplots()
            ax.matshow(cat_masks)
            for (i, j), z in np.ndenumerate(cat_masks):
                ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
            
            img_save_path = os.path.join(self.log_dir, 'masks.png')
            fig.savefig(img_save_path, bbox_inches='tight')
            plt.close(fig)
            
            if self.summary_writer is not None:
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
            
            print("Epoch {}".format(epoch))
            print("att {} - class loss: {:.2f} div loss: {:.2f} l1 loss: {:.2f} total loss: {:.2f} acc: {:.2f}".format(idx, 
                                                                    classifier_losses[idx]/counter,
                                                                    div_losses[idx]/counter,
                                                                    l1_losses[idx]/counter,
                                                                    loss[idx]/counter,
                                                                    classifier_acc[idx]/counter))

            logger.info("Epoch {}".format(epoch))
            logger.info("att {} - class loss: {:.2f} div loss: {:.2f} l1 loss: {:.2f} total loss: {:.2f} acc: {:.2f}".format(idx, 
                                                                    classifier_losses[idx]/counter,
                                                                    div_losses[idx]/counter,
                                                                    l1_losses[idx]/counter,
                                                                    loss[idx]/counter,
                                                                    classifier_acc[idx]/counter))
    def vote(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            x = self.embedding.feature_extractor()
            conf = self.confidences.weights(False)
            pred_y = self.classifier(x)
        
        votes = torch.argmax(pred_y, axis=-1)
        
        self.votes = votes
        
        return pred_y, votes, conf
        
    def get_attentions(self):
        return self.classifier.get_attention_masks()

    def update_confidence(self,
                          was_successful: bool,
                          votes: list):
        success_count = votes
        failure_count = np.ones(len(votes)) - votes
        
        if not was_successful:
            success_count = failure_count
            failure_count = votes
        
        self.confidences.update_successes(success_count)
        self.confidences.update_failures(failure_count)






