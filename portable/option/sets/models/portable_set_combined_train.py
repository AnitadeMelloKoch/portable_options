import torch
import torch.nn as nn
import torch.optim as optim
import portable.option.ensemble.criterion as criterion
import numpy as np
import os

from portable.option.sets.models import MLP, SmallEmbedding
from portable.option.sets.utils import BayesianWeighting
import logging

logger = logging.getLogger(__name__)

class EnsembleClassifier():

    def __init__(self, 
        device,
        embedding_output_size=64, 
        learning_rate=1e-4,
        num_modules=8, 
        num_output_classes=2,
        batch_k=8,
        normalize=False,
        
        beta_distribution_alpha=30,
        beta_distribution_beta=5,
        margin=1):
        
        self.num_modules = num_modules
        self.num_output_classes = num_output_classes
        self.device = device

        self.embedding = SmallEmbedding(embedding_size=embedding_output_size, 
                                   num_attention_modules=self.num_modules,
                                   batch_k=batch_k,
                                   normalize=normalize).to(self.device)
        self.classifiers = nn.ModuleList([
            MLP(embedding_output_size, self.num_output_classes) for _ in range(
                self.num_modules)]).to(self.device)
        
        self.optimizer = optim.Adam(
            list(self.embedding.parameters()) + list(self.classifiers.parameters()),
            learning_rate
        )

        self.avg_loss = np.zeros(num_modules)
        self.classifier_loss = nn.CrossEntropyLoss()

        self.confidences = BayesianWeighting(
            beta_distribution_alpha,
            beta_distribution_beta,
            self.num_modules,
            device
        )
        self.margin = margin

    def save(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.embedding.state_dict(), os.path.join(path, 'embedding.pt'))
        torch.save(self.classifiers.state_dict(), os.path.join(path, 'classifier.pt'))
        self.confidences.save(os.path.join(path, 'confidence'))
        np.save(os.path.join(path, 'loss.npy'), self.avg_loss)

    def load(self, path):

        if not os.path.exists(os.path.join(path, 'embedding.pt')):
            return

        if not os.path.exists(os.path.join(path, 'classifier.pt')):
            return

        if not os.path.exists(os.path.join(path, 'confidence')):
            return

        if os.path.exists(os.path.join(path, 'loss.npy')):
            self.avg_loss = np.load(os.path.join(path, 'loss.npy'))
        else:
            print("NO LOSS FILE FOUND")

        print('Loading from {}'.format(path))

        self.embedding.load_state_dict(torch.load(os.path.join(path, 'embedding.pt')))
        self.classifiers.load_state_dict(torch.load(os.path.join(path, 'classifier.pt')))
        self.confidences.load(os.path.join(path, 'confidence'))

    def set_classifiers_train(self):
        for classifier in self.classifiers:
            classifier.train()
    
    def set_classifiers_eval(self):
        for classifier in self.classifiers:
            classifier.eval()

    def train(self, 
              dataset, 
              epochs):
        
        self.set_classifiers_train()
        self.embedding.train()
        
        for epoch in range(epochs):
            class_loss = 0
            class_loss_record = np.zeros(self.num_modules)
            class_acc = np.zeros(self.num_modules)
            div_loss = 0
            loss_record = 0
            
            num_batches = dataset.num_batches
            dataset.shuffle()
            for _ in range(num_batches):
                loss = 0
                x, y = dataset.get_batch(shuffle_batch=True)
                max_x = torch.max(x)
                if max_x >1:
                    x /= 255
                x = x.to(self.device)
                y = y.to(self.device)
                
                embeddings = self.embedding(x)
                l_div = criterion.batched_L_divergence(embeddings, self.confidences.weights(), self.margin)
                loss += l_div
                # print('l_div:', l_div)
                l_class = 0
                for idx in range(self.num_modules):
                    attention_x = embeddings[:, idx, :]
                    pred_y = self.classifiers[idx](attention_x)
                    cl_loss = self.classifier_loss(pred_y, y)
                    l_class += cl_loss
                    class_loss_record[idx] += cl_loss.item()
                    class_loss += cl_loss
                    pred_classes = torch.argmax(pred_y, dim=1).detach()
                    class_acc[idx] += (torch.sum(pred_classes==y).item()/len(x))
                
                # print('l_class:', l_class/self.num_modules)
                loss += l_class
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_record += loss.item()
                div_loss += l_div.item()
            
            div_loss /= num_batches
            class_loss_record /= num_batches
            class_acc /= num_batches
            loss_record /= num_batches
            
            logger.info("Epoch {}: Total loss = {} divergence loss = {}".format(epoch, loss_record, div_loss))
            print("Epoch {}: Total loss = {} divergence loss = {}".format(epoch, loss_record, div_loss))
            for idx in range(self.num_modules):
                logger.info("\tClassifier {}: loss {:.4f} accuracy {:.4f}".format(idx, class_loss_record[idx], class_acc[idx]))
                print("\tClassifier {}: loss {:.4f} accuracy {:.4f}".format(idx, class_loss_record[idx], class_acc[idx]))

    def get_votes(self, x, return_attention=False):
        self.set_classifiers_eval()
        self.embedding.eval()
        
        max_x = torch.max(x)
        if max_x >1:
            x /= 255

        x = x.to(self.device)

        embeddings = self.embedding(x, return_attention_mask=False).detach()
        pred_idx = np.zeros(self.num_modules, dtype=np.int16)
        pred = np.zeros(self.num_modules)
        for idx in range(self.num_modules):
            attention_x = embeddings[:, idx, :]
            pred_y = self.classifiers[idx](attention_x).detach().cpu().numpy()[0]
            pred_idx[idx] = np.argmax(pred_y)
            pred[idx] = pred_y[pred_idx[idx]]

        if return_attention:
            return pred_idx, pred, self.confidences.weights(False), self.get_attention(x)

        return pred_idx, pred, self.confidences.weights(False)

    def get_single_module(self, x, module):
        self.set_classifiers_eval()
        self.embedding.eval()
        
        max_x = torch.max(x)
        if max_x > 1:
            with torch.no_grad():
                x/=255
        x = x.to(self.device)
        embedding = self.embedding.forward_one_attention(x, module).squeeze()

        return self.classifiers[module](embedding)


    def update_successes(self, successes):
        self.confidences.update_successes(successes)

    def update_failures(self, failures):
        self.confidences.update_failures(failures)

    def get_attention(self, x):
        self.set_classifiers_eval()
        self.embedding.eval()
        x = x.to(self.device)
        _, atts = self.embedding(x, return_attention_mask=True)
        return atts
