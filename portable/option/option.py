import logging
import os
import pickle
from enum import IntEnum

from portable.option.sets import EnsembleClassifier
from portable.option.memory import SetDataset
from portable.option.sets import PositionSetPair

logger = logging.getLogger(__name__)

class actions(IntEnum):
    INVALID         = -1
    NOOP            = 0
    FIRE            = 1
    UP              = 2
    RIGHT           = 3
    LEFT            = 4
    DOWN            = 5
    UP_RIGHT        = 6
    UP_LEFT         = 7
    DOWN_RIGHT      = 8
    DOWN_LEFT       = 9
    UP_FIRE         = 10
    RIGHT_FIRE      = 11
    LEFT_FIRE       = 12
    DOWN_FIRE       = 13
    UP_RIGHT_FIRE   = 14
    UP_LEFT_FIRE    = 15
    DOWN_RIGHT_FIRE = 16
    DOWN_LEFT_FIRE  = 17

class Skill():
    # Base class for skills
    def __init__(
        self,
        device,
        vote_function,

        problem_termination_epsilon=3,

        initiation_attention_module_num=8,
        initiation_votes_needed=1,
        initiation_embedding_learning_rate=1e-4,
        initiation_classifier_learning_rate=1e-2,
        initiation_embedding_output_size=64,

        termination_attention_module_num=8,
        termination_votes_needed=1,
        termination_embedding_learning_rate=1e-4,
        termination_classifier_learning_rate=1e-2,
        termination_embedding_output_size=64,

        max_tries=20
    
    ):
        
        self.initiation_classifier = EnsembleClassifier(
            device=device,
            num_votes_needed=initiation_votes_needed,
            embedding_learning_rate=initiation_embedding_learning_rate,
            classifier_learning_rate=initiation_classifier_learning_rate,
            num_modules=initiation_attention_module_num,
            embedding_output_size=initiation_embedding_output_size
        )
        self.termination_classifier = EnsembleClassifier(
            device=device,
            num_votes_needed=termination_votes_needed,
            embedding_learning_rate=termination_embedding_learning_rate,
            classifier_learning_rate=termination_classifier_learning_rate,
            num_modules=termination_attention_module_num,
            embedding_output_size=termination_embedding_output_size
        )

        self.max_tries = max_tries
        self.steps = 0
        self.executing_skill = False
        self.vote_function = vote_function

        self.initiation_dataset = SetDataset()
        self.termination_dataset = SetDataset()

        self.problem_space_classifiers = []

        self.problem_termination_epsilon = problem_termination_epsilon

    def add_local_classifier(self, obs, info, termination_image, termination_point):
        self.problem_space_classifiers.append(PositionSetPair(
            obs,
            info,
            termination_image,
            termination_point
        ))

    def check_local_termination(self, pos, idx):
        return self.problem_space_classifiers[idx].check_termination(pos)

    def check_local_initiation(self, pos):
        idx = 0
        for pair in self.problem_space_classifiers:
            prediction = pair.check_initiation(pos)
            if prediction == 1:
                return True, idx
            idx += 1

        return False, -1

    def add_negative_trajectories_initiation(self, obs, info, classifier_idx):
        if classifier_idx == -1:
            return 

        self.problem_space_classifiers[classifier_idx].add_negative(
                obs,
                info
            )

    def add_positive_trajectories_initiation(self, obs, info, classifier_idx):
        if classifier_idx == -1:
            return

        self.problem_space_classifiers[classifier_idx].add_positive(
            obs,
            info
        )

    def start_skill(self):
        self.executing_skill = True
        self.steps = 0

    def end_skill(self):
        self.executing_skill = False

    def save(self, path):
        self.initiation_classifier.save(os.path.join(path, 'initiation'))
        self.termination_classifier.save(os.path.join(path, 'termination'))

        filename = os.path.join(path, 'markov_pairs.pkl')
        if len(self.problem_space_classifiers) != 0:
            with open(filename, "wb") as f:
                pickle.dump(self.problem_space_classifiers, f)

    def load(self, path):
        self.initiation_classifier.load(os.path.join(path, 'initiation'))
        self.termination_classifier.load(os.path.join(path, 'termination'))

        filename = os.path.join(path, 'markov_pairs.pkl')
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.problem_space_classifiers = pickle.load(f)

    def check_global_initiation(self, x):
        votes, vote_conf = self.initiation_classifier.get_votes(x)
        vote = self.vote_function(votes, vote_conf)

        return votes, vote_conf, vote

    def check_global_termination(self, x):
        votes, vote_conf = self.termination_classifier.get_votes(x)
        vote = self.vote_function(votes, vote_conf)

        return votes, vote_conf, vote        

    def train_initiation(self, 
            num_cycles=1, 
            embedding_epochs_per_cycle=10, 
            classifier_epochs_per_cycle=10):
        for i in range(num_cycles):
            self.initiation_classifier.train_embedding(self.initiation_dataset, embedding_epochs_per_cycle)
            self.initiation_classifier.train_classifiers(self.initiation_dataset, classifier_epochs_per_cycle)

    def train_termination(self,
            num_cycles=1, 
            embedding_epochs_per_cycle=10, 
            classifier_epochs_per_cycle=10):
        for i in range(num_cycles):

            self.termination_classifier.train_embedding(self.termination_dataset, embedding_epochs_per_cycle)
            self.termination_classifier.train_classifiers(self.termination_dataset, classifier_epochs_per_cycle)

    def add_data_from_file_initiation(self, positive_files, negative_files):
        self.initiation_dataset.add_true_files(positive_files)
        self.initiation_dataset.add_false_files(negative_files)

    def add_data_from_file_termination(self, positive_files, negative_files):
        self.termination_dataset.add_true_files(positive_files)
        self.termination_dataset.add_false_files(negative_files)

    def get_action(self, agent_state):
        pass
