import logging
import numpy as np
import torch
from portable.option.memory import SetDataset
from portable.option.sets.models import EnsembleClassifier

class Set():
    def __init__(
            self,
            device,
            vote_function,

            attention_module_num=8,
            embedding_learning_rate=1e-4,
            classifier_learning_rate=1e-2,
            embedding_output_size=64,
            beta_distribution_alpha=30,
            beta_distribution_beta=5,
            stack_size=4,

            dataset_max_size=100000,
            dataset_batch_size=16
        ):
        self.classifier = EnsembleClassifier(
            device=device,
            stack_size=stack_size,
            embedding_learning_rate=embedding_learning_rate,
            classifier_learning_rate=classifier_learning_rate,
            num_modules=attention_module_num,
            embedding_output_size=embedding_output_size,
            batch_k=dataset_batch_size//2,
            beta_distribution_alpha=beta_distribution_alpha,
            beta_distribution_beta=beta_distribution_beta
        )

        self.vote_function = vote_function

        self.dataset = SetDataset(max_size=dataset_max_size, batchsize=dataset_batch_size)

        self.attention_module_num = attention_module_num
        self.votes = None
        self.avg_loss = np.zeros(self.attention_module_num)

    def save(self, path):
        self.classifier.save(path)
        self.dataset.save(path)

    def load(self, path):
        self.classifier.load(path)
        self.dataset.load(path)
        self.avg_loss = self.classifier.avg_loss

    def add_data(   
            self,
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

    def add_data_from_files(
            self,
            positive_files,
            negative_files,
            priority_negative_files):
        assert isinstance(positive_files, list)
        assert isinstance(negative_files, list)
        assert isinstance(priority_negative_files, list)

        self.dataset.add_true_files(positive_files)
        self.dataset.add_false_files(negative_files)
        self.dataset.add_priority_false_files(priority_negative_files)

    def loss(   
            self,
            positive_samples,
            negative_samples):
        assert isinstance(positive_samples, list)
        assert isinstance(negative_samples, list)

        dataset = SetDataset()
        if len(positive_samples) > 0:
            dataset.add_true_data(positive_samples)
        if len(negative_samples) > 0:
            dataset.add_false_data(negative_samples)

        loss, accuracy = self.classifier.get_loss(dataset)

        return loss

    def train(
            self,
            embedding_epochs,
            classifier_epochs):
            
        self.classifier.train(
            self.dataset,
            embedding_epochs,
            classifier_epochs
        )

        self.avg_loss = self.classifier.avg_loss

    def vote(
            self,
            agent_state):

        agent_state = torch.unsqueeze(agent_state, dim=0)

        votes, conf, confidences = self.classifier.get_votes(agent_state)
        vote = self.vote_function(votes, confidences, conf)

        logger.info("[set] Classifier votes: {}  \n\tConfidences: {}, \n\tWeights: {}, \n\tFinal vote: {}".format(votes, conf, confidences, vote))

        self.votes = votes

        return vote

    def get_attentions(self, agent_state):
        agent_state = torch.unsqueeze(agent_state, dim=0)
        votes, conf, confidences, attentions = self.classifier.get_votes(agent_state, True)

        return votes, conf, confidences, attentions

    def update_confidence(
            self,
            was_successful,
            votes):
        success_count = votes
        failure_count = np.ones(len(votes)) - votes

        if not was_successful:
            success_count = failure_count
            failure_count = votes

        self.classifier.update_successes(success_count)
        self.classifier.update_failures(failure_count)
