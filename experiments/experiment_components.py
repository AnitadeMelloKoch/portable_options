import logging
import datetime
import os
import random
import gin
import torch
import lzma
import dill
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from portable.option import Option
from portable.utils import set_player_ram, plot_attentions, plot_state
from portable.option.sets.utils import get_vote_function, VOTE_FUNCTION_NAMES
from portable.environment.agent_wrapper import actions

@gin.configurable
class PolicyExperiment():
    def __init__(self,
                 base_dir,
                 seed,
                 experiment_name,
                 policy_phi,
                 device_type="cpu",
                 policy_success_rate=0.9):

        assert device_type in ["cpu", "cuda"]

        self.device = torch.device(device_type)
        self.option = Option(
            device=self.device,
            initiation_vote_function=get_vote_function(VOTE_FUNCTION_NAMES[0]),
            termination_vote_function=get_vote_function(VOTE_FUNCTION_NAMES[0]),
            policy_phi=policy_phi
        )
        random.seed(seed)
        self.seed = seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, self.name, str(self.seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(
            filename=log_file,
            format='%(asctime)s %(levelname)s: %(message)s',
            level=logging.INFO
        )

        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        

        self.policy_success_rate = policy_success_rate

        self.trial_data = {}

        self.room_num = 0

    def save(self):
        file_name = os.path.join(self.save_dir, 'trial_data.pkl')
        with lzma.open(file_name, 'wb') as f:
            dill.dump(self.trial_data, f)

    def load(self):
        file_name = os.path.join(self.save_dir, 'trial_data.pkl')
        if os.path.exists(file_name):
            with lzma.open(file_name, 'rb') as f:
                self.trial_data = dill.load(f)

    def run_trial(self,
                  env,
                  max_steps):
        self.trial_data[self.room_num] = {}
        self.trial_data["rewards"] = []
        self.trial_data["steps"] = []

        step_number, total_reward, steps, rewards = self.option.bootstrap_policy(
            env,
            max_steps,
            self.policy_success_rate
        )

        self.trial_data[self.room_num]["rewards"] = rewards
        self.trial_data[self.room_num]["steps"] = steps

        self.room_num += 1

        self.save()


@gin.configurable
class ClassifierExperiment():
    def __init__(self,
                 base_dir,
                 seed,
                 experiment_name,
                 options_initiation_positive_files,
                 options_initiation_negative_files,
                 options_initiation_priority_negative_files,
                 train_initiation_embedding_epochs,
                 train_initiation_classifier_epochs,
                 options_termination_positive_files,
                 options_termination_negative_files,
                 options_termination_priority_negative_files,
                 train_termination_embedding_epochs,
                 train_termination_classifier_epochs,
                 device_type="cpu"):

        assert device_type in ["cpu", "cuda"]

        self.device = torch.device(device_type)
        self.option = Option(
            device=self.device,
            initiation_vote_function=get_vote_function(VOTE_FUNCTION_NAMES[0]),
            termination_vote_function=get_vote_function(VOTE_FUNCTION_NAMES[0]),
            policy_phi=lambda x: x
        )
        random.seed(seed)
        self.seed = seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, self.name, str(self.seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(
            filename=log_file,
            format='%(asctime)s %(levelname)s: %(message)s',
            level=logging.INFO
        )

        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        
        self.trial_data = {}

        self.trial_data["x"] = []
        self.trial_data["y"] = []
        self.trial_data["room"] = []
        self.trial_data["initiation_votes"] = []
        self.trial_data["termination_votes"] = []

        self._train_initiation(
            options_initiation_positive_files,
            options_initiation_negative_files,
            options_initiation_priority_negative_files,
            train_initiation_embedding_epochs,
            train_initiation_classifier_epochs
        )

        self._train_termination(
            options_termination_positive_files,
            options_termination_negative_files,
            options_termination_priority_negative_files,
            train_termination_embedding_epochs,
            train_termination_classifier_epochs
        )

    def _train_initiation(self,
                          positive_files,
                          negative_files,
                          priority_negative_files,
                          embedding_epochs,
                          classifier_epochs):

        
        if len(positive_files) == 0:
            logging.warning('[experiment] No positive files were given for the initiation set.')

        if len(negative_files) == 0:
            logging.warning('[experiment] No negative files were given for the initiation set.')

        self.option.initiation.add_data_from_files(
            positive_files,
            negative_files,
            priority_negative_files
        )
        self.option.initiation.train(
            embedding_epochs,
            classifier_epochs
        )

    def _train_termination(self,
                           positive_files,
                           negative_files,
                           priority_negative_files,
                           embedding_epochs,
                           classifier_epochs):
        
        if len(positive_files) == 0:
            logging.warning('[experiment] No positive files were given for the termination set.')

        if len(negative_files) == 0:
            logging.warning('[experiment] No negative files were given for the termination set.')

        self.option.termination.add_data_from_files(
            positive_files,
            negative_files,
            priority_negative_files
        )
        self.option.termination.train(
            embedding_epochs,
            classifier_epochs
        )

    def save(self):
        file_name = os.path.join(self.save_dir, 'trial_data.pkl')
        with lzma.open(file_name, 'wb') as f:
            dill.dump(self.trial_data, f)

    def load(self):
        file_name = os.path.join(self.save_dir, 'trial_data.pkl')
        if os.path.exists(file_name):
            with lzma.open(file_name, 'rb') as f:
                self.trial_data = dill.load(f)

    def perform_action(self, action, step_num, env, info):
        for _ in range(step_num):
            # fig = plt.figure(num=1, clear=True)
            # ax = fig.add_subplot()
            initiation_vote = self.option.initiation.vote(info["stacked_agent_state"])
            self.trial_data["initiation_votes"].append(np.sum(self.option.initiation.votes))
            termination_vote = self.option.termination.vote(info["stacked_agent_state"])
            self.trial_data["termination_votes"].append(np.sum(self.option.termination.votes))
            x, y, room = info["position"]
            self.trial_data["x"].append(x)
            self.trial_data["y"].append(y)
            self.trial_data["room"].append(room)

            obs, reward, done, info = env.step(action)

            # screen = env.render('rgb_array')
            # ax.imshow(screen)
            # plt.show(block=False)
            # plt.pause(3)

        return info

    def plot(self, base_img):

        base_img = np.flipud(base_img)

        self.trial_data["x"] = [x + 2 for x in self.trial_data["x"]]
        self.trial_data["y"] = [y - 116 for y in self.trial_data["y"]]

        init = plt.figure(num=1, clear=True)
        init_ax = init.add_subplot()
        init_ax.imshow(base_img)
        init_ax.invert_yaxis()
        init_ax.axis("off")
        im = init_ax.scatter(self.trial_data["x"], self.trial_data["y"], c=self.trial_data["initiation_votes"],
            marker='x',
            alpha=0.5
        )
        init.colorbar(im, ax=init_ax)
        init.savefig(os.path.join(self.plot_dir, "init_votes_room_{}.png".format(self.trial_data["room"][0])), bbox_inches='tight')

        term = plt.figure(num=1, clear=True)
        term_ax = term.add_subplot()
        term_ax.imshow(base_img)
        term_ax.invert_yaxis()
        term_ax.axis("off")
        im = term_ax.scatter(self.trial_data["x"], self.trial_data["y"], c=self.trial_data["termination_votes"],
            marker='x',
            alpha=0.5
        )
        term.colorbar(im, ax=term_ax)
        term.savefig(os.path.join(self.plot_dir, "term_votes_room_{}.png".format(self.trial_data["room"][0])), bbox_inches='tight')



    def to_right_room(self,
                  env):
        
        obs, info = env.reset()

        base_img = env.render('rgb_array')

        info = info = self.perform_action(actions.LEFT, 2, env, info)
        info = self.perform_action(actions.RIGHT, 4, env, info)
        info = self.perform_action(actions.LEFT, 2, env, info)
        info = self.perform_action(actions.RIGHT, 2, env, info)
        info = self.perform_action(actions.RIGHT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 8, env, info)
        info = self.perform_action(actions.RIGHT, 7, env, info)
        info = self.perform_action(actions.LEFT, 7, env, info)
        info = self.perform_action(actions.LEFT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 8, env, info)
        info = self.perform_action(actions.LEFT, 4, env, info)
        info = self.perform_action(actions.LEFT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 8, env, info)
        info = self.perform_action(actions.LEFT, 6, env, info)
        info = self.perform_action(actions.RIGHT, 6, env, info)
        info = self.perform_action(actions.RIGHT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 8, env, info)
        info = self.perform_action(actions.RIGHT, 2, env, info)
        info = self.perform_action(actions.DOWN, 10, env, info)
        info = self.perform_action(actions.RIGHT, 8, env, info)
        info = self.perform_action(actions.RIGHT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 6, env, info)
        info = self.perform_action(actions.UP, 6, env, info)
        info = self.perform_action(actions.DOWN, 3, env, info)
        info = self.perform_action(actions.RIGHT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 6, env, info)
        info = self.perform_action(actions.RIGHT, 6, env, info)
        info = self.perform_action(actions.LEFT, 6, env, info)
        info = self.perform_action(actions.LEFT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 6, env, info)
        info = self.perform_action(actions.DOWN, 3, env, info)
        info = self.perform_action(actions.LEFT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 6, env, info)
        info = self.perform_action(actions.LEFT, 2, env, info)
        info = self.perform_action(actions.UP, 10, env, info)
        info = self.perform_action(actions.RIGHT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 12, env, info)
        info = self.perform_action(actions.DOWN, 1, env, info)
        info = self.perform_action(actions.RIGHT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 6, env, info)
        info = self.perform_action(actions.RIGHT, 1, env, info)
        info = self.perform_action(actions.DOWN, 10, env, info)
        info = self.perform_action(actions.LEFT, 7, env, info)
        info = self.perform_action(actions.NOOP, 4, env, info)
        info = self.perform_action(actions.LEFT, 1, env, info)
        info = self.perform_action(actions.NOOP, 3, env, info)
        info = self.perform_action(actions.LEFT, 1, env, info)
        info = self.perform_action(actions.NOOP, 3, env, info)
        info = self.perform_action(actions.LEFT, 1, env, info)
        info = self.perform_action(actions.NOOP, 3, env, info)
        info = self.perform_action(actions.LEFT, 1, env, info)
        info = self.perform_action(actions.NOOP, 3, env, info)
        info = self.perform_action(actions.LEFT, 1, env, info)
        info = self.perform_action(actions.NOOP, 3, env, info)
        info = self.perform_action(actions.LEFT, 1, env, info)
        info = self.perform_action(actions.NOOP, 3, env, info)
        info = self.perform_action(actions.LEFT, 1, env, info)
        info = self.perform_action(actions.NOOP, 3, env, info)
        info = self.perform_action(actions.LEFT, 1, env, info)
        info = self.perform_action(actions.NOOP, 3, env, info)
        info = self.perform_action(actions.NOOP, 5, env, info)
        info = self.perform_action(actions.LEFT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 9, env, info)
        info = self.perform_action(actions.LEFT, 8, env, info)
        info = self.perform_action(actions.UP, 10, env, info)
        info = self.perform_action(actions.LEFT, 2, env, info)
        info = self.perform_action(actions.UP_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 10, env, info)
        info = self.perform_action(actions.RIGHT, 2, env, info)
        info = self.perform_action(actions.DOWN, 10, env, info)
        info = self.perform_action(actions.RIGHT, 13, env, info)
        info = self.perform_action(actions.RIGHT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 9, env, info)
        info = self.perform_action(actions.RIGHT, 10, env, info)
        info = self.perform_action(actions.UP, 10, env, info)
        info = self.perform_action(actions.LEFT, 2, env, info)
        info = self.perform_action(actions.LEFT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 6, env, info)
        info = self.perform_action(actions.DOWN, 3, env, info)
        info = self.perform_action(actions.LEFT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 6, env, info)
        info = self.perform_action(actions.LEFT, 2, env, info)
        info = self.perform_action(actions.UP, 10, env, info)
        info = self.perform_action(actions.RIGHT, 2, env, info)
        info = self.perform_action(actions.RIGHT_FIRE, 1, env, info)
        info = self.perform_action(actions.NOOP, 9, env, info)
        info = self.perform_action(actions.RIGHT, 50, env, info)
        info = self.perform_action(actions.LEFT, 19, env, info)
        info = self.perform_action(actions.DOWN, 40, env, info)
        info = self.perform_action(actions.LEFT, 30, env, info)
        info = self.perform_action(actions.RIGHT, 60, env, info)

        self.plot(base_img)

        self.save()
