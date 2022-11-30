import argparse
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

from portable.option import Option
from portable.utils import set_player_ram
from portable.option.sets.utils import get_vote_function, VOTE_FUNCTION_NAMES
from portable.option.policy.agents.abstract_agent import evaluating

@gin.configurable
class Experiment():
    def __init__(
        self,
        base_dir,
        seed,
        experiment_name,
        initiation_vote_function,
        termination_vote_function,
        policy_phi,
        experiment_env_function,
        get_percentage_function,
        check_termination_true_function,
        device_type="cpu",
        train_initiation=True,
        initiation_positive_files=[],
        initiation_negative_files=[],
        initiation_priority_negative_files=[],
        train_initiation_embedding_epoch_per_cycle=10,
        train_initiation_classifier_epoch_per_cycle=10,
        train_initiation_cycles=1,
        train_termination=True,
        termination_positive_files=[],
        termination_negative_files=[],
        termination_priority_negative_files=[],
        train_termination_embedding_epoch_per_cycle=10,
        train_termination_classifier_epoch_per_cycle=10,
        train_termination_cycles=1,
        train_policy=True,
        policy_bootstrap_env=None,
        train_policy_max_steps=10000,
        train_policy_success_rate=0.8,
        max_option_tries=5
        ):

        assert device_type in ["cpu", "cuda"]
        assert initiation_vote_function in VOTE_FUNCTION_NAMES
        assert termination_vote_function in VOTE_FUNCTION_NAMES

        self.device = torch.device(device_type)

        self.option = Option(
            device=self.device,
            initiation_vote_function=get_vote_function(initiation_vote_function),
            termination_vote_function=get_vote_function(termination_vote_function),
            policy_phi=policy_phi
        )

        random.seed(seed)
        self.seed = seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, self.name, str(self.seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)

        self.env = experiment_env_function(self.seed)

        self._get_percent_completed = get_percentage_function
        self._check_termination_correct = check_termination_true_function
        self.max_option_tries = max_option_tries

        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(
            filename=log_file,
            format='%(asctime)s %(levelname)s: %(message)s',
            level=logging.INFO
        )

        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        
        self.load()

        if train_initiation:
            self._train_initiation(
                initiation_positive_files,
                initiation_negative_files,
                initiation_priority_negative_files,
                train_initiation_embedding_epoch_per_cycle,
                train_initiation_classifier_epoch_per_cycle,
                train_initiation_cycles
            )
        
        if train_termination:
            self._train_termination(
                termination_positive_files,
                termination_negative_files,
                termination_priority_negative_files,
                train_termination_embedding_epoch_per_cycle,
                train_termination_classifier_epoch_per_cycle,
                train_termination_cycles
            )

        if train_policy:
            self._train_policy(
                policy_bootstrap_env,
                train_policy_max_steps,
                train_policy_success_rate
            )

        self.trial_data = {
            "name": [],
            "performance": [],
            "start_pos": [],
            "end_pos": [],
            "true_terminations": [],
            "completed": [],
            "dead": []
        }


    def _train_initiation(
            self,
            positive_files,
            negative_files,
            priority_negative_files,
            embedding_epochs_per_cycle,
            classifier_epochs_per_cycle,
            number_cycles):

        if len(positive_files) == 0:
            logging.warning('[experiment] No positive files were given for the initiation set.')

        if len(negative_files) == 0:
            logging.warning('[experiment] No negative files were given for the initiation set.')

        self.option.add_data_from_files_initiation(
            positive_files,
            negative_files,
            priority_negative_files
        )
        self.option.train_initiation(
            embedding_epochs_per_cycle,
            classifier_epochs_per_cycle,
            number_cycles,
            shuffle_data=True
        )

    def _train_termination(
            self,
            positive_files,
            negative_files,
            priority_negative_files,
            embedding_epochs_per_cycle,
            classifier_epochs_per_cycle,
            number_cycles):

        if len(positive_files) == 0:
            logging.warning('[experiment] No positive files were given for the termination set.')

        if len(negative_files) == 0:
            logging.warning('[experiment] No negative files were given for the termination set.')

        self.option.add_data_from_files_termination(
            positive_files,
            negative_files,
            priority_negative_files
        )
        self.option.train_termination(
            embedding_epochs_per_cycle,
            classifier_epochs_per_cycle,
            number_cycles,
            shuffle_data=True
        )

    def _train_policy(
            self,
            bootstrap_env,
            max_steps,
            success_rate_for_well_trained):
        self.option.bootstrap_policy(
            bootstrap_env,
            max_steps,
            success_rate_for_well_trained
        )

    def save(self, additional_path=None):
        if additional_path is not None:
            save_dir = os.path.join(self.save_dir, additional_path)
        else:
            save_dir = self.save_dir
        # self.option.save(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        file_name = os.path.join(self.save_dir, 'trial_data.pkl')
        with lzma.open(file_name, 'wb') as f:
            dill.dump(self.trial_data, f)

    def load(self, additional_path=None):
        if additional_path is not None:
            save_dir = os.path.join(self.save_dir, additional_path)
        else:
            save_dir = self.save_dir
        self.option.load(save_dir)
        file_name = os.path.join(self.save_dir, 'trial_data.pkl')
        if os.path.exists(file_name):
            with lzma.open(file_name, 'rb') as f:
                self.trial_data = dill.load(f)

    def _set_env_ram(self, ram, state, agent_state, use_agent_space):
        self.env.reset()
        _ = set_player_ram(self.env, ram)
        self.env.stacked_state = state
        self.env.stacked_agent_state = agent_state

        if use_agent_space:
            return agent_state
        else:
            return state

    def run_trial(
            self, 
            possible_inits,
            true_terminations,
            number_episodes_in_trial,
            eval, 
            trial_name="",
            use_agent_space=False):
        assert isinstance(possible_inits, list)
        logging.info("[experiment] Starting trial {}".format(trial_name))
        print("[experiment] Starting trial {}".format(trial_name))

        results = []
        start_poses = []
        end_poses = []
        true_terminationses = []
        completeds = []
        deads = []

        for x in range(number_episodes_in_trial):
            logging.info("Episode {}/{}".format(x, number_episodes_in_trial))
            completed = False
            
            rand_idx = random.randint(0, len(possible_inits)-1)
            rand_state = possible_inits[rand_idx]
            state = self._set_env_ram(
                rand_state["ram"],
                rand_state["state"],
                rand_state["agent_state"],
                use_agent_space
            )

            agent_state = rand_state["agent_state"]
            start_pos = rand_state["position"]
            # print(rand_state["position"])
            info = self.env.get_current_info({})
            attempt = 0
            timedout = 0
            must_break = False
            while (attempt < self.max_option_tries) and (timedout < 3) and (not must_break):
            # for y in range(self.max_option_tries):
                attempt += 1
                logging.info("Attempt {}/{}".format(attempt, self.max_option_tries))
                can_initiate = self.option.can_initiate(agent_state, (info["player_x"],info["player_y"]))

                if not can_initiate:
                    must_break = True
                else:
                    if eval:
                        _, _, done, info, _ = self.option.evaluate(
                            self.env,
                            state,
                            info
                        )
                    else:
                        _, _, done, info, _ = self.option.run(
                            self.env,
                            state,
                            info,
                            [0],
                            0
                        )

                    if info["needs_reset"]:
                        must_break = True

                    if info['option_timed_out']:
                        logging.info("[experiment] option timed out {} times".format(timedout))
                        timedout += 1

                    if done:
                        must_break = True

                    agent_state = info["stacked_agent_state"]
                    position = info["position"]

                    completed = self._check_termination_correct(position, true_terminations[rand_idx], self.env)
                    if completed:
                        # print("termination triggered")
                        must_break = True
            # print(position)
            result = self._get_percent_completed(start_pos, position, true_terminations[rand_idx], self.env)
            if info["dead"]:
                result = 0
            # print('result',result)
            # input("press any button to continue")
            results.append(result)
            start_poses.append(start_pos)
            end_poses.append(position)
            true_terminationses.append(true_terminations[rand_idx])
            completeds.append(completed)
            deads.append(info["dead"])
            logging.info("Result: {}".format(completed))
            
        self.trial_data["name"].append(trial_name)
        self.trial_data["start_pos"].append(start_poses)
        self.trial_data["end_pos"].append(end_poses)
        self.trial_data["true_terminations"].append(true_terminationses)
        self.trial_data["completed"].append(completeds)
        self.trial_data["dead"].append(deads)

        if not eval:
            self.option.train_initiation( 5, 10)
            self.option.train_termination( 5, 10)

        logging.info("[experiment] Finished trial {} performance: {}".format(
            trial_name,
            np.mean(results)
            ))
        print("[experiment] Finished trial {} performance: {}".format(
            trial_name,
            np.mean(results)
            ))
        
    def bootstrap_from_room(
            self,
            possible_inits,
            true_terminations, 
            number_episodes_in_trial,
            use_agent_space=False):
        
        assert isinstance(possible_inits, list)
        assert isinstance(true_terminations, list)
        
        logging.info("Bootstrapping weights from training room")
        print("Bootstrapping weights from training room")

        for x in range(number_episodes_in_trial):
            logging.info("Episode {}/{}".format(x, number_episodes_in_trial))
            rand_idx = random.randint(0, len(possible_inits)-1)
            rand_state = possible_inits[rand_idx]
            state = self._set_env_ram(
                rand_state["ram"],
                rand_state["state"],
                rand_state["agent_state"],
                use_agent_space
            )

            agent_state = rand_state["agent_state"]
            info = self.env.get_current_info({})
            done = False

            count = 0
            timedout = 0

            while (not done) and (info["needs_reset"]) and (count < 100) and (timedout < 3):

                count += 1

                can_initiate = self.option.can_initiate(agent_state, info)

                if not can_initiate:
                    logging.info("Break because initiation was not triggered")
                    break

                state, _, done, info, _ = self.option.evaluate(
                    self.env,
                    state,
                    info
                )

                if info['needs_reset']:
                    logging.info("Breaking because environment needs reset")
                    break

                if info['option_timed_out']:
                    logging.info('[experiment] option has timed out {} times'.format(timedout))
                    timedout += 1

                if self._check_termination_correct(self.env.get_current_position(), true_terminations[rand_idx], self.env):
                    self.option.termination_update_confidence(was_successful=True)
                    logging.info("Breaking because correct termination was found")
                    break
                else:
                    self.option.termination_update_confidence(was_successful=False)

    def plot(self, names):

        results = {}
        for x in range(len(names)):
            results[names[x]] = []
        
        for x in range(len(self.trial_data["name"])):
            trial_name = self.trial_data["name"][x]
            if trial_name.find("bootstrap") == -1:
                trial_room = trial_name.split("_")[0]
                results[trial_room].append(np.mean(self.trial_data["performance"][x]))

        for key in results:
            plt.plot(results[key])

        plt.savefig(self.log_dir + '/plot.jpg')






