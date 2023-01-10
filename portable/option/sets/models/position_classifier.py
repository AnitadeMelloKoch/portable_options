import random
import itertools
import numpy as np  
from collections import deque
from sklearn.svm import OneClassSVM, SVC
import os
import pickle

class TrainingExample:
    def __init__(self, obs, pos):
        self.obs = obs
        self.x = pos[0]
        self.y = pos[1]

    @property
    def pos(self):
        pos = self.x, self.y
        return np.array(pos)

    def __iter__(self):
        return ((self.obs, self.x, self.y) for _ in [0])

class PositionClassifier:
    def __init__(self, epsilon=None, gamma=1, maxlen=600):
        self.classifier = None
        self.positive_examples = deque([], maxlen=maxlen)
        self.negative_examples = deque([], maxlen=maxlen)
        self.epsilon = epsilon
        self.gamma = gamma

    def save(self, path):
        if self.classifier is not None:
            with open(os.path.join(path, 'classifier.pkl')) as f:
                pickle.dump(self.classifier, f)
        
        if len(self.positive_examples) > 0:
            with open(os.path.join(path, 'positive_examples.pkl')) as f:
                pickle.dump(self.positive_examples, f)
        
        if len(self.negative_examples) > 0:
            with open(os.path.join(path, 'negative_examples.pkl')) as f:
                pickle.dump(self.negative_examples)

        np.save(os.path.join(path, 'epsilon.npy'), self.epsilon)
        np.save(os.path.join(path, 'gamma.npy'), self.gamma)

    def load(self, path):
        classifier_file = os.path.join(path, 'classifier.pkl')
        positive_file = os.path.join(path, 'positive_examples.pkl')
        negative_file = os.path.join(path, 'negative_examples.pkl')

        if os.path.exists(classifier_file):
            with open(classifier_file, "rb") as f:
                self.classifier = pickle.load(f)

        if os.path.exists(positive_file):
            with open(positive_file, "rb") as f:
                self.positive_examples = pickle.load(f)

        if os.path.exists(negative_file):
            with open(negative_file, "rb") as f:
                self.negative_examples = pickle.load(f)

        if os.path.exists(os.path.join(path, 'epsilon.npy')):
            self.epsilon = np.load(os.path.join(path, 'epsilon.npy'))

        if os.path.exists(os.path.join(path, 'gamma.npy')):
            self.gamma = np.load(os.path.join(path, 'gamma.npy'))

    def predict(self, state):
        assert isinstance(self.classifier, (OneClassSVM, SVC))
        return self.classifier.predict([state])[0] == 1

    def is_initialized(self):
        return self.classifier is not None

    def add_positive_examples(self, images, positions):
        if len(images) != len(positions):
            print('length images', len(images))
            print('length positions', len(positions))
        assert len(images) == len(positions)

        positive_examples = [TrainingExample(img, pos) for img, pos in zip(images, positions)]
        self.positive_examples.append(positive_examples)

    def add_negative_examples(self, images, positions):
        if len(images) != len(positions):
            print('length images', len(images))
            print('length positions', len(positions))
        assert len(images) == len(positions)

        negative_examples = [TrainingExample(img, pos) for img, pos in zip(images, positions)]
        self.negative_examples.append(negative_examples)

    def add_examples_from_epsilon_square(self, img, info):
        assert self.epsilon is not None
        assert len(info) == 1
        info = info[0]
        
        position = (info['player_x'], info['player_y'])

        def in_epsilon_square(current_position):
            if current_position[0] <= (position[0]+self.epsilon) and \
                current_position[0] >= (position[0]-self.epsilon) and \
                current_position[1] <= (position[1]+self.epsilon) and \
                current_position[1] >= (position[1]-self.epsilon):
                return True
            return False   

        positions_neg = []
        positions_pos = []
        imgs_neg = []
        imgs_pos = []

        x_start, x_end = int(position[0]-(1.75)*self.epsilon), int(position[0]+(1.75)*self.epsilon)+1
        y_start, y_end = int(position[1]-(1.75)*self.epsilon), int(position[1]+(1.75)*self.epsilon)+1

        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                current_position = (x,y)
                if in_epsilon_square(current_position):
                    positions_pos.append(current_position)
                    imgs_pos.append(img)
                else:
                    positions_neg.append(current_position)
                    imgs_neg.append(img)

        self.add_positive_examples(imgs_pos, positions_pos)
        self.add_negative_examples(imgs_neg, positions_neg)


    @staticmethod
    def construct_feature_matrix(examples):
        examples = list(itertools.chain.from_iterable(examples))
        positions = [example.pos for example in examples]
        return np.array(positions)

    def get_images(self):
        positive_samples = [example.obs for example in self.positive_examples]
        negative_samples = [example.obs for example in self.negative_examples]

        return positive_samples, negative_samples

    def fit_classifier(self):
        if (len(self.negative_examples) > 0 and len(self.positive_examples) > 0):
            self.train_two_class_classifier()
        elif len(self.positive_examples) > 0:
            self.train_one_class_svm()

    def train_one_class_svm(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        self.classifier = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        self.classifier.fit(positive_feature_matrix)

    def train_two_class_classifier(self):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        
        positive_labels = [1]*positive_feature_matrix.shape[0]
        negative_labels = [0]*negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))

        # if negative_feature_matrix.shape[0] >= 10:
        #     kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight":"balanced"}
        # else:
        #     kwargs = {"kernel": "rbf", "gamma": "scale"}

        kwargs = {"kernel": "rbf", "gamma": self.gamma, "class_weight":"balanced"}
        

        self.classifier = SVC(**kwargs)
        self.classifier.fit(X, Y)

    def sample(self, epsilon=2.):
        # Epsilon-safe sampling from the pessimistic initiation classifier.

        def compile_states(s):
            # Get positions that lie in an epsilon-box around s.pos
            pos0 = s.pos
            pos1 = np.copy(pos0)
            pos2 = np.copy(pos0)
            pos3 = np.copy(pos0)
            pos4 = np.copy(pos0)
            pos1[0] -= epsilon
            pos2[0] += epsilon
            pos3[1] -= epsilon
            pos4[1] += epsilon

            return pos0, pos1, pos2, pos3, pos4

        idxs = list(range(len(self.positive_examples)))
        random.shuffle(idxs)

        for idx in idxs:
            sampled_trajectory = self.positive_examples[idx]

            positions = []
            for s in sampled_trajectory:
                positions.extend(compile_states(s))

            position_matrix = np.vstack(positions)
            predictions = self.pessimistic_classifier.predict(position_matrix) == 1
            predictions = np.reshape(predictions, (-1, 5))
            valid = np.all(predictions, axis=1)
            indices = np.argwhere(valid == True)

            if len(indices) > 0:
                return sampled_trajectory[indices[0][0]]

        return self.sample_from_initiation_region()

    def sample_from_initiation_region(self):
        # Sample from pessimistic initiation classifier

        num_tries = 0
        sampled_state = None
        while sampled_state is None and num_tries < 200:
            num_tries = num_tries + 1
            sampled_trajectory_idx = random.choice(range(len(self.positive_examples)))
            sampled_trajectory = self.positive_examples[sampled_trajectory_idx]
            sampled_state = self.get_first_state_in_classifier(sampled_trajectory)
        return sampled_state

    def get_first_state_in_classifier(self, trajectory):
        # Extract first state in trajectory that is inside initiation classifier

        for state in trajectory:
            assert isinstance(state, TrainingExample)
            if self.pessimistic_predict(state.pos):
                return state

    def get_states_inside_pessimistic_classifier_region(self):
        def get_observations(idx):
            positive_examples = itertools.chain.from_iterable(self.positive_examples)
            return [positive_examples[i].obs for i in idx]

        if self.pessimistic_classifier is not None:
            point_array = self.construct_feature_matrix(self.positive_examples)
            point_array_predictions = self.pessimistic_classifier.predict(point_array)
            positive_indices = np.where(point_array_predictions==1)
            positive_observations = get_observations(positive_indices)
            return positive_observations
        return []