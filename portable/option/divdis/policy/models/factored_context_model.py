import numpy as np
from collections import deque 
from sklearn.svm import OneClassSVM

class FactoredContextClassifier():
    def __init__(self, maxlen=100):
        self.classifier = None
        self.positive_examples = deque([], maxlen=maxlen)
    
    def predict(self, obs):
        assert isinstance(self.classifier, OneClassSVM)
        return self.classifier.predict([obs])[0] == 1
    
    def is_initialized(self):
        return self.classifier is not None
    
    def add_positive_examples(self, obs_list):
        self.positive_examples.append(obs_list)
    
    def fit(self, nu=0.1):
        positive_feature_matrix = np.array(self.positive_examples)
        self.classifier = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        self.classifier.fit(positive_feature_matrix)