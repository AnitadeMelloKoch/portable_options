import torch
import torch.nn as nn
import torch.optim as optim
import portable.option.ensemble.criterion as criterion
import numpy as np
import os

from portable.option.sets.models import MLP
from portable.option.ensemble import Attention
from portable.option.sets.utils import BayesianWeighting
import logging

logger = logging.getLogger(__name__)

class EnsembleClassifier():

    def __init__(self, 
        device,
        embedding_output_size=64, 
        embedding_learning_rate=1e-4,
        classifier_learning_rate=1e-2,
        num_modules=8, 
        num_output_classes=2,
        
        beta_distribution_alpha=30,
        beta_distribution_beta=5):
        
        self.num_modules = num_modules
        self.num_output_classes = num_output_classes
        self.device = device

        self.embedding = Attention(embedding_size=embedding_output_size, 
                                   num_attention_modules=self.num_modules).to(self.device)
        self.classifiers = nn.ModuleList([
            MLP(embedding_output_size, self.num_output_classes) for _ in range(
                self.num_modules)]).to(self.device)

        self.classifier_optimizer = optim.Adam(
            list(self.classifiers.parameters()),
            classifier_learning_rate
        )

        self.embedding_optimizer = optim.SGD(
            list(self.embedding.parameters()),
            embedding_learning_rate,
            momentum=0.95,
            weight_decay=1e-4
        )

        self.avg_loss = np.zeros(num_modules)
        self.classifier_loss = nn.CrossEntropyLoss()

        self.confidences = BayesianWeighting(
            beta_distribution_alpha,
            beta_distribution_beta,
            self.num_modules,
            device
        )

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

    def set_train(self):
        self.embedding.train()
        for classifier in self.classifiers:
            classifier.train()
    
    def set_eval(self):
        self.embedding.eval()
        for classifier in self.classifiers:
            classifier.eval()

    def train(self, dataset, epochs):
        self.set_train()
        for epoch in range(epochs):
            dataset.shuffle()
            num_batches = dataset.num_batches
            avg_loss = 0
            avg_accuracy = np.zeros(self.num_modules)
            for num in range(num_batches):
                loss = 0
                x, y = dataset.get_batch()
                x = x.to(self.device)
                y = y.to(self.device)

                embeddings = self.embedding(x)
                l_div = criterion.batched_criterion(embeddings, y, self.confidences.weights())
                for idx, classifier in enumerate(self.classifiers):
                    classifier_input = embeddings[:,idx,:]
                    pred_y = classifier(classifier_input)
                    c_loss = self.classifier_loss(pred_y, y)
                    loss += self.confidences.weights()[idx]*c_loss
                    pred_class = torch.argmax(pred_y, dim=1).detach()
                    avg_accuracy[idx] += (torch.sum(pred_class == y).item()/len(x))
                
                loss = loss/self.num_modules
                loss += l_div

                self.classifier_optimizer.zero_grad()
                self.embedding_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()
                self.embedding_optimizer.step()
                avg_loss += loss.item()
                
            avg_loss = avg_loss/num_batches
            avg_accuracy = avg_accuracy/num_batches

            logger.info("Epoch {}: Avg loss - {}".format(epoch, loss))
            # maybe print weighting
            for idx in range(self.num_modules):
                logger.info("  Classifier {}: accuracy {:.4f}".format(idx, avg_accuracy[idx]))

            print("Epoch {}: Avg loss = {}".format(epoch, loss))
            # maybe print weighting
            for idx in range(self.num_modules):
                print("  Classifier {}: accuracy {:.4f}".format(idx, avg_accuracy[idx]))


        self.set_eval()
        self.avg_loss = avg_loss

    def get_votes(self, x, return_attention=False):
        self.set_eval()
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

    def update_successes(self, successes):
        self.confidences.update_successes(successes)

    def update_failures(self, failures):
        self.confidences.update_failures(failures)

    def get_attention(self, x):
        self.set_eval()
        x = x.to(self.device)
        _, atts = self.embedding(x, return_attention_mask=True)
        return atts
