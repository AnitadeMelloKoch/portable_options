import random
import itertools
import numpy as np
from collections import deque
from pfrl.wrappers import atari_wrappers
from sklearn.svm import OneClassSVM, SVC
from portable.option.divdis.policy.models.svm_evaluator import SVMEvaluator

class InitiationClassifier:
    def __init__(self, optimistic_classifier, pessimistic_classifier):
        self.optimistic_classifier = optimistic_classifier
        self.pessimistic_classifier = pessimistic_classifier

    def optimistic_predict(self, state):
        pass

    def pessimistic_predict(self, state):
        pass

    def add_positive_examples(self, trajectory):
        pass

    def add_negative_examples(self, trajectory):
        pass

    def sample(self):
        pass

class FactoredInitiationClassifier(InitiationClassifier):
    def __init__(self, maxlen=100):
        optimistic_classifier = None
        pessimistic_classifier = None
        self.positive_examples = deque([], maxlen=maxlen)
        self.negative_examples = deque([], maxlen=maxlen)
        self.evaluator = SVMEvaluator()
        super().__init__(optimistic_classifier, pessimistic_classifier)

    def optimistic_predict(self, state):
        assert isinstance(self.optimistic_classifier, (OneClassSVM, SVC))
        return self.optimistic_classifier.predict([state.numpy()])[0] == 1

    def pessimistic_predict(self, state):
        print(self.pessimistic_classifier)
        assert isinstance(self.pessimistic_classifier, (OneClassSVM, SVC))
        return self.pessimistic_classifier.predict([state.numpy()])[0] == 1
    
    def is_initialized(self):
        return self.optimistic_classifier is not None and \
            self.pessimistic_classifier is not None

    def get_false_positive_rate(self):  # TODO: Implement this
        return np.array([0., 0.])

    def add_positive_examples(self, obs_list):
        obs_list = [x.numpy() for x in obs_list]
        self.positive_examples.extend(obs_list)

    def add_negative_examples(self, obs_list):
        obs_list = [x.numpy() for x in obs_list]
        self.negative_examples.extend(obs_list)

    @staticmethod
    def construct_feature_matrix(examples):
        # examples = list(itertools.chain.from_iterable(examples))
        # positions = [example.pos for example in examples]
        return np.array(examples)

    def fit(self):
        if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
            self.train_two_class_classifier()
        elif len(self.positive_examples) > 0:
            self.train_one_class_svm()

    def train_one_class_svm(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        self.pessimistic_classifier.fit(positive_feature_matrix)

        self.optimistic_classifier = OneClassSVM(kernel="rbf", nu=nu/10., gamma="scale")
        self.optimistic_classifier.fit(positive_feature_matrix)

    def train_two_class_classifier(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))

        if negative_feature_matrix.shape[0] >= 10:
            kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
        else:
            kwargs = {"kernel": "rbf", "gamma": "scale"}

        self.optimistic_classifier = SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y)

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
            self.pessimistic_classifier.fit(positive_training_examples)

    def evaluate_svm(self):
        
        
        self.evaluator.plot_surface
        pass
    
    