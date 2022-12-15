from json import load
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import os

class BayesianWeighting():

    def __init__(self,
        alpha,
        beta,
        num_heads):
        self.alpha = alpha
        self.beta = beta
        self.num_heads = num_heads

        self.weights = np.ones(self.num_heads)

        self.successes = np.zeros(self.num_heads)
        self.failures = np.zeros(self.num_heads)

        self.update_weights()

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, 'alpha.npy'), self.alpha)
        np.save(os.path.join(path, 'beta.npy'), self.beta)
        np.save(os.path.join(path, 'failure_count.npy'), self.failures)
        np.save(os.path.join(path, 'success_count.npy'), self.successes)

    def load(self, path, load_previous_alpha_beta=True):
        if not os.path.exists(os.path.join(path, 'success_count.npy')):
            print('No success count found. Returning')
            return

        if not os.path.exists(os.path.join(path, 'failure_count.npy')):
            print('No failure count found. Returning')
            return

        if load_previous_alpha_beta and not os.path.exists(os.path.join(path, 'alpha.npy')):
            print('No alpha found but alpha was supposed to be loaded. Returning')
            return

        if load_previous_alpha_beta and not os.path.exists(os.path.join(path, 'beta.npy')):
            print('No beta found but beta was supposed to be loaded. Returning')
            return

        self.successes = np.load(os.path.join(path, 'success_count.npy'))
        self.failures = np.load(os.path.join(path, 'failure_count.npy'))

        if load_previous_alpha_beta:
            self.alpha = np.load(os.path.join(path, 'alpha.npy'))
            self.beta = np.load(os.path.join(path, 'beta.npy'))

        self.update_weights()

    def update_successes(self, successes):
        # e.g. if you have 4 classifiers should have input [0, 2, 1, 3]
        # where classifier 0 has succeeded 0 times
        #       classifier 1 has succeeded 2 times
        #       classifier 2 has succeeded 1 time
        #       classifier 3 has succeeded 3 times
        assert len(successes) == self.num_heads
        assert all(x >= 0 for x in successes)

        self.successes = self.successes + successes
        self.update_weights()

    def update_failures(self, failures):
        # e.g. if you have 4 classifiers should have input [3, 1, 2, 0]
        # where classifier 0 has failed 3 times
        #       classifier 1 has failed 1 time
        #       classifier 2 has failed 2 times
        #       classifier 3 has failed 0 times
        assert len(failures) == self.num_heads
        assert all(x >= 0 for x in failures)

        self.failures = self.failures + failures
        self.update_weights()

    def update_weights(self):
        x = np.linspace(0, 1, 1000)
        for idx in range(self.num_heads):
            posterior = beta.pdf(x, self.alpha+self.successes[idx],
                                        self.beta+self.failures[idx])
            self.weights[idx] = x[np.argmax(posterior)]

    def plot_posterior(self, idx):
        x = np.linspace(0, 1, 1000)
        posterior = beta.pdf(x, self.alpha+self.successes[idx],
                            self.beta+self.failures[idx])
        
        plt.plot(x,posterior)
        plt.show()

