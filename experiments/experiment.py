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
        device_type="cpu",
        train_initiation=True,
        initiation_positive_files=[],
        initiation_negative_files=[],
        train_initiation_embedding_epoch_per_cycle=10,
        train_initiation_classifier_epoch_per_cycle=10,
        train_initiation_cycles=1,
        train_termination=True,
        termination_positive_files=[],
        termination_negative_files=[],
        train_termination_embedding_epoch_per_cycle=10,
        train_termination_classifier_epoch_per_cycle=10,
        train_termination_cycles=1,
        train_policy=True,
        policy_bootstrap_env=None,
        train_policy_max_steps=10000,
        train_policy_success_rate=0.8
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


        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(
            filename=log_file,
            format='%(asctime)s %(levelname)s: %(message)s',
            level=logging.INFO
        )

        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        
        if train_initiation:
            self._train_initiation(
                initiation_positive_files,
                initiation_negative_files,
                train_initiation_embedding_epoch_per_cycle,
                train_initiation_classifier_epoch_per_cycle,
                train_initiation_cycles
            )
        
        if train_termination:
            self._train_termination(
                termination_positive_files,
                termination_negative_files,
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
            "performance": []
        }


    def _train_initiation(
            self,
            positive_files,
            negative_files,
            embedding_epochs_per_cycle,
            classifier_epochs_per_cycle,
            number_cycles):

        if len(positive_files) == 0:
            logging.warning('[experiment] No positive files were given for the initiation set.')

        if len(negative_files) == 0:
            logging.warning('[experiment] No negative files were given for the initiation set.')

        self.option.add_data_from_files_initiation(
            positive_files,
            negative_files
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
            embedding_epochs_per_cycle,
            classifier_epochs_per_cycle,
            number_cycles):

        if len(positive_files) == 0:
            logging.warning('[experiment] No positive files were given for the termination set.')

        if len(negative_files) == 0:
            logging.warning('[experiment] No negative files were given for the termination set.')

        self.option.add_data_from_files_termination(
            positive_files,
            negative_files
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

    def save(self):
        self.option.save(self.save_dir)
        file_name = os.path.join(self.save_dir, 'trial_data.pkl')
        with lzma.open(file_name, 'wb') as f:
            dill.dump(self.trial_data, f)

    def load(self):
        self.option.load(self.save_dir)
        file_name = os.path.join(self.save_dir, 'trial_data.pkl')
        if os.path.exists(file_name):
            with lzma.open(file_name, 'rb') as f:
                self.trial_data = dill.load(f)

    def _set_env_ram(self, ram, state, agent_state, use_agent_space):
        _ = set_player_ram(self.env, ram)
        self.env.stacked_state = state
        self.env.stacked_agent_state = agent_state

        if use_agent_space:
            return agent_state
        else:
            return state

    @staticmethod
    def _get_percent_completed(start_pos, final_pos, possible_terminations):
        def manhatten(a, b):
            return sum(abs(val1, val2) for val1, val2 in zip((a[0],a[1]),(b[0],b[1])))

        original_distance = []
        completed_distance = []
        for term in possible_terminations:
            original_distance.append(start_pos, term)
            completed_distance.append(final_pos, term)
        original_distance = np.mean(original_distance)
        completed_distance = np.mean(completed_distance)

        return 1 - completed_distance/original_distance

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

        self.env.reset()

        results = []

        for _ in range(number_episodes_in_trial):
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
            info = self.env.get_current_info({})

            can_initiate = self.option.can_initiate(agent_state, info)

            if not can_initiate:
                results.append(0)
            else:
                if eval:
                    _, _, _, _, _ = self.option.evaluate(
                        self.env,
                        state,
                        info
                    )
                else:
                    _, _, _, _, _ = self.option.run(
                        self.env,
                        state,
                        info,
                        [0],
                        0
                    )

                results.append(self._get_percent_completed(start_pos, self.env.get_current_position(), true_terminations))
            
        self.trial_data["name"].append(trial_name)
        self.trial_data["performance"].append(results)

        if not eval:
            self.option.train_initiation( 0, 50)
            self.option.train_termination( 0, 50)

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
        
        self.env.reset()

        for _ in range(number_episodes_in_trial):
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

            while not done or info["needs_reset"]:

                logging.info("info: {}".format(info))

                can_initiate = self.option.can_initiate(agent_state, info)

                if not can_initiate:
                    break

                state, _, done, info, _ = self.option.evaluate(
                    self.env,
                    state,
                    info
                )

                if self.env.get_current_position() in true_terminations[rand_idx]:
                    self.option.termination_update_confidence(was_successful=True)
                    break
                else:
                    self.option.termination_update_confidence(was_successful=False)









