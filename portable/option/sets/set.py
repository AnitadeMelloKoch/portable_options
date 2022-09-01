import numpy as np
from portable.option.memory import SetDataset
from portable.option.sets.utils import BayesianWeighting
from portable.option.sets.models import EnsembleClassifier

class Set():
    def __init__(
            self,
            device,
            vote_function,

            beta_distribution_alpha=20,
            beta_distribution_beta=10,

            attention_module_num=8,
            embedding_learning_rate=1e-4,
            classifier_learning_rate=1e-2,
            embedding_output_size=64,

            dataset_max_size=100000
        ):
        self.classifier = EnsembleClassifier(
            device=device,
            embedding_learning_rate=embedding_learning_rate,
            classifier_learning_rate=classifier_learning_rate,
            num_modules=attention_module_num,
            embedding_output_size=embedding_output_size
        )

        self.vote_function = vote_function

        self.dataset = SetDataset(max_size=dataset_max_size)

        self.confidence = BayesianWeighting(
            beta_distribution_alpha,
            beta_distribution_beta,
            attention_module_num
        )

    def save(self, path):
        self.classifier.save(path)
        self.confidence.save(path)
        self.dataset.save(path)

    def load(self, path):
        self.classifier.load(path)
        self.confidence.load(path)
        self.dataset.load(path)

    def add_data(   
            self,
            positive_data=[],
            negative_data=[]):
        assert isinstance(positive_data, list)
        assert isinstance(negative_data, list)

        if len(positive_data) > 0:
            self.dataset.add_true_data(positive_data)
        
        if len(negative_data) > 0:
            self.dataset.add_false_data(negative_data)

    def add_data_from_files(
            self,
            positive_files,
            negative_files):
        assert isinstance(positive_files, list)
        assert isinstance(negative_files, list)

        self.dataset.add_true_files(positive_files)
        self.dataset.add_false_files(negative_files)

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
            num_cycles=1,
            embedding_epochs_per_cycle=10,
            classifier_epochs_per_cycle=10):
        for i in range(num_cycles):
            self.classifier.train_embedding(
                self.dataset,
                embedding_epochs_per_cycle
            )
            self.classifier.train_classifiers(
                self.dataset,
                classifier_epochs_per_cycle
            )

    def vote(
            self,
            agent_state):
        votes, conf = self.classifier.get_votes(agent_state)
        vote = self.vote_function(votes, self.confidence.weights, conf)

        return vote

    def update_confidence(
            self,
            success_count,
            failure_count):
        self.confidence.update_successes(success_count)
        self.confidence.update_failures(failure_count)
