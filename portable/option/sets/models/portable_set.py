import torch
import torch.nn as nn
import torch.optim as optim
import portable.option.ensemble.criterion as criterion
import numpy as np
import os

from portable.option.sets.models import MLP, SmallEmbedding
import logging

logger = logging.getLogger(__name__)

class EnsembleClassifier():

    def __init__(self, 
        device,
        embedding_output_size=64, 
        embedding_learning_rate=1e-4, 
        classifier_learning_rate=1e-2, 
        num_modules=8, 
        batch_k=8, 
        normalize=False, 
        num_output_classes=2):
        
        self.num_modules = num_modules
        self.batch_k = batch_k
        self.normalize = normalize
        self.num_output_classes = num_output_classes
        self.device = device

        self.embedding = SmallEmbedding(embedding_size=embedding_output_size, num_attention_modules=self.num_modules, batch_k=self.batch_k, normalize=self.normalize).to(self.device)
        self.classifiers = nn.ModuleList([MLP(embedding_output_size, self.num_output_classes) for _ in range(self.num_modules)]).to(self.device)

        self.embedding_optimizer = optim.SGD(self.embedding.parameters(), embedding_learning_rate, momentum=0.95, weight_decay=1e-4)
        self.classifiers_optimizers = []
        for i in range(self.num_modules):
            optimizer = optim.Adam(self.classifiers[i].parameters(), classifier_learning_rate)
            self.classifiers_optimizers.append(optimizer)

        self.avg_loss = np.zeros(num_modules)

    def save(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.embedding.state_dict(), os.path.join(path, 'embedding.pt'))
        torch.save(self.classifiers.state_dict(), os.path.join(path, 'classifier.pt'))
        np.save(os.path.join(path, 'loss.npy'), self.avg_loss)

    def load(self, path):

        if not os.path.exists(os.path.join(path, 'embedding.pt')):
            return

        if not os.path.exists(os.path.join(path, 'classifier.pt')):
            return

        if os.path.exists(os.path.join(path, 'loss.npy')):
            self.avg_loss = np.load(os.path.join(path, 'loss.npy'))
        else:
            print("NO LOSS FILE FOUND")

        print('Loading from {}'.format(path))

        self.embedding.load_state_dict(torch.load(os.path.join(path, 'embedding.pt')))
        self.classifiers.load_state_dict(torch.load(os.path.join(path, 'classifier.pt')))

    def set_classifiers_train(self):
        for i in range(self.num_modules):
            self.classifiers[i].train()

    def set_classifiers_eval(self):
        for i in range(self.num_modules):
            self.classifiers[i].eval()

    def train_embedding(self, dataset, epochs, shuffle_data):
        # dataset is a pytorch dataset
        self.embedding.train()
        self.set_classifiers_eval()

        for epoch in range(epochs):
            loss_div, loss_homo, loss_heter = 0, 0, 0
            counter = 0
            num_batches = dataset.num_batches
            if shuffle_data:
                dataset.shuffle()
            for _ in range(num_batches + 1):
                x, _ = dataset.get_batch()
                x = x.to(self.device)
                _, anchors, positives, negatives, _ = self.embedding(x, sampling=True, return_attention_mask=False)
                if anchors.size()[0] == 0:
                    continue
                anchors = anchors.view(anchors.size(0), self.num_modules, -1)
                positives = positives.view(positives.size(0), self.num_modules, -1)
                negatives = negatives.view(negatives.size(0), self.num_modules, -1)

                self.embedding_optimizer.zero_grad()
                l_div, l_homo, l_heter = criterion.criterion(anchors, positives, negatives)
                l = l_div + l_homo + l_heter
                
                l.backward()
                self.embedding_optimizer.step()

                loss_homo += l_homo.item()
                loss_heter += l_heter.item()
                loss_div += l_div.item()

                counter += 1

            loss_homo /= (counter+1)
            loss_heter /= (counter+1)
            loss_div /= (counter+1)

            logger.info('Epoch %d batches %d\tdiv:%.4f\thomo:%.4f\theter:%.4f'%(epoch, counter+1, loss_div, loss_homo, loss_heter))

        self.embedding.eval()
        self.classifiers.eval()

    def train_classifiers(self, dataset, epochs, shuffle_data):
        self.embedding.eval()
        self.set_classifiers_train()

        for epoch in range(epochs):
            avg_loss = np.zeros(self.num_modules)
            avg_accuracy = np.zeros(self.num_modules)
            count = 0
            num_batches = dataset.num_batches
            if shuffle_data:
                dataset.shuffle()
            for _ in range(num_batches):
                x, y = dataset.get_batch(shuffle=True)
                x = x.to(self.device)
                y = y.to(self.device)
                embeddings = self.embedding(x, sampling=False, return_attention_mask=False)
                for idx in range(self.num_modules):
                    attention_x = embeddings[:,idx,:]
                    pred_y = self.classifiers[idx](attention_x)
                    classifier_criterion = nn.CrossEntropyLoss()
                    loss = classifier_criterion(pred_y, y)
                    self.classifiers_optimizers[idx].zero_grad()
                    loss.backward(retain_graph=True)
                    self.classifiers_optimizers[idx].step()
                    avg_loss[idx] += loss.item()
                    pred_classes = torch.argmax(pred_y, dim=1).detach()
                    avg_accuracy[idx] += (torch.sum(pred_classes == y).item()/len(x))
                count += 1
            avg_loss = avg_loss/count
            avg_accuracy = avg_accuracy/count

            logger.info("Epoch {}:".format(epoch))
            for idx in range(self.num_modules):
                logger.info("\t - Classifier {}: loss {:.4f} accuracy {:.4f}".format(idx, avg_loss[idx], avg_accuracy[idx]))
            logger.info("Average across classifiers: loss = {:.4f} accuracy = {:.4f}".format(np.mean(avg_loss), np.mean(avg_accuracy)))
        self.embedding.eval()
        self.classifiers.eval()

        self.avg_loss = avg_loss

    def get_votes(self, x):
        self.embedding.eval()
        self.set_classifiers_eval()

        x = x.to(self.device)

        embeddings = self.embedding(x, sampling=False, return_attention_mask=False).detach()
        pred_idx = np.zeros(self.num_modules, dtype=np.int16)
        pred = np.zeros(self.num_modules)

        for idx in range(self.num_modules):
            attention_x = embeddings[:, idx, :]
            pred_y = self.classifiers[idx](attention_x).detach().cpu().numpy()[0]
            pred_idx[idx] = np.argmax(pred_y)
            pred[idx] = pred_y[pred_idx[idx]]

        return pred_idx, pred

    def get_attention(self, x):
        self.embedding.eval()
        x = x.to(self.device)

        _, atts = self.embedding(x, sampling=False, return_attention_mask=True)

        return atts

    def get_loss(self, dataset):
        avg_loss = np.zeros(self.num_modules)
        avg_accuracy = np.zeros(self.num_modules)
        count = 0
        num_batches = dataset.num_batches
        for _ in range(num_batches):
            x, y = dataset.get_batch(shuffle=True)
            x = x.to(self.device)
            y = y.to(self.device)
            embeddings = self.embedding(x, sampling=False, return_attention_mask=False)
            for idx in range(self.num_modules):
                attention_x = embeddings[:,idx,:]
                pred_y = self.classifiers[idx](attention_x)
                classifier_criterion = nn.CrossEntropyLoss()
                avg_loss[idx] += classifier_criterion(pred_y, y).item()
                pred_classes = torch.argmax(pred_y, dim=1).detach()
                avg_accuracy[idx] += (torch.sum(pred_classes == y).item()/len(x))
            count += 1
        avg_loss = avg_loss/count
        avg_accuracy = avg_accuracy/count

        return avg_loss, avg_accuracy



