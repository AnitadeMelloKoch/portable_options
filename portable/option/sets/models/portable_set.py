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
        embedding_learning_rate=1e-4,
        classifier_learning_rate=1e-2,
        num_modules=8, 
        num_output_classes=2,
        batch_k=8,
        normalize=False,
        
        beta_distribution_alpha=30,
        beta_distribution_beta=5):
        
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

        self.classifier_optimizers = []
        for i in range(self.num_modules):
            optimizer = optim.Adam(self.classifiers[i].parameters(), classifier_learning_rate)
            self.classifier_optimizers.append(optimizer)
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

    def set_classifiers_train(self):
        for classifier in self.classifiers:
            classifier.train()
    
    def set_classifiers_eval(self):
        for classifier in self.classifiers:
            classifier.eval()

    def train(self, 
              dataset, 
              embedding_epochs, 
              classifier_epochs):
        self.train_embedding(dataset, embedding_epochs)
        self.train_classifiers(dataset, classifier_epochs)

    def train_embedding(self, dataset, epochs):
        self.set_classifiers_eval()
        self.embedding.train()

        for epoch in range(epochs):
            loss_div, loss_homo, loss_heter = 0,0,0
            counter = 0
            num_batches = dataset.num_batches
            dataset.shuffle()
            for _ in range(num_batches):
                x, _ = dataset.get_batch(shuffle_batch=False)
                x = x.to(self.device)

                _, anchors, positive, negative, _ = self.embedding(x, sampling=True)
                if anchors.size()[0] == 0:
                    continue
                
                anchors = anchors.view(anchors.size(0), self.num_modules, -1)
                positive = positive.view(positive.size(0), self.num_modules, -1)
                negative = negative.view(negative.size(0), self.num_modules, -1)

                l_div, l_homo, l_heter = criterion.criterion(anchors, positive, negative, self.confidences.weights())
                loss = l_div + l_homo + l_heter

                self.embedding_optimizer.zero_grad()
                loss.backward()
                self.embedding_optimizer.step()

                loss_div += l_div.item()
                loss_homo += l_homo.item()
                loss_heter += l_heter.item()

                counter += 1

            loss_homo /= counter+1
            loss_heter /= counter+1
            loss_div /= counter+1

            logger.info('Epoch {}: \tdiv:{:.4f}\thomo:{:.4f}\theter:{:.4f}'.format(epoch, loss_homo, loss_heter, loss_div))
            print('Epoch {}: \tdiv:{:.4f}\thomo:{:.4f}\theter:{:.4f}'.format(epoch, loss_homo, loss_heter, loss_div))

    def train_classifiers(self, dataset, epochs):
        self.embedding.eval()
        self.set_classifiers_train()

        for epoch in range(epochs):
            avg_loss = np.zeros(self.num_modules)
            avg_accuracy = np.zeros(self.num_modules)
            count = 0
            num_batches = dataset.num_batches
            dataset.shuffle()
            for _ in range(num_batches):
                x, y = dataset.get_batch()
                x = x.to(self.device)
                y = y.to(self.device)

                embeddings = self.embedding(x)
                for idx in range(self.num_modules):
                    attention_x = embeddings[:,idx,:]
                    pred_y = self.classifiers[idx](attention_x)
                    loss = self.classifier_loss(pred_y, y)
                    self.classifier_optimizers[idx].zero_grad()
                    loss.backward(retain_graph=True)
                    self.classifier_optimizers[idx].step()
                    avg_loss[idx] += loss.item()
                    pred_classes = torch.argmax(pred_y, dim=1).detach()
                    avg_accuracy[idx] += (torch.sum(pred_classes==y).item()/len(x))
                count += 1
            avg_loss = avg_loss/count
            avg_accuracy = avg_accuracy/count

            logger.info("Epoch {}:".format(epoch))
            print("Epoch {}:".format(epoch))
            for idx in range(self.num_modules):
                logger.info("\tClassifier {}: loss {:.4f} accuracy {:.4f}".format(idx, avg_loss[idx], avg_accuracy[idx]))
                print("\tClassifier {}: loss {:.4f} accuracy {:.4f}".format(idx, avg_loss[idx], avg_accuracy[idx]))


    def get_votes(self, x, return_attention=False):
        self.set_classifiers_eval()
        self.embedding.eval()
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
        self.set_classifiers_eval()
        self.embedding.eval()
        x = x.to(self.device)
        _, atts = self.embedding(x, return_attention_mask=True)
        return atts
