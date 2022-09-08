import argparse
import logging
import datetime
import os
import random
import gin
import torch
import lzma
import dill

from portable.option import Option
from portable.utils import set_player_ram, load_gin_configs
from portable.option.sets.utils import get_vote_function, VOTE_FUNCTION_NAMES

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
            number_cycles
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
            number_cycles
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


    def run_trial(
            self, 
            possible_inits,
            number_episodes_in_trial, 
            trial_name="",
            use_agent_space=False):
        assert isinstance(possible_inits, list)
        logging.info("[experiment] Starting trial {}".format(trial_name))

        results = []

        for _ in range(number_episodes_in_trial):
            rand_state = random.choice(possible_inits)
            state = self._set_env_ram(
                rand_state["ram"],
                rand_state["state"],
                rand_state["agent_state"],
                use_agent_space
            )
            agent_state = rand_state["agent_state"]
            info = self.env.get_current_info()

            can_initiate = self.option.can_initiate(agent_state, info)

            if not can_initiate:
                results.append(0)
            else:
                _, total_reward, _, _, steps = self.option.run(
                    self.env,
                    state,
                    info,
                    [0],
                    0
                )

                results.append(total_reward)
            
        self.trial_data["name"].append(trial_name)
        self.trial_data["performance"].append(results)

        logging.info("[experiment] Finished trial {} average reward: {}".format(
            trial_name,
            sum(results)/number_episodes_in_trial
            ))
        
        