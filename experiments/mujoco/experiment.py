import datetime
import torch
import os
import random
import logging
from pfrl.utils import set_random_seed
import lzma
import dill
import gin

from experiments.mujoco.environment.vec_env import VecExtractDictObs, VecNormalize, VecChannelOrder, VecMonitor
from experiments.mujoco.environment import make_ant_env
from portable.option import MujocoOption

@gin.configurable
class MujocoExperiment():

    def __init__(
            self,
            base_dir,
            seed,
            experiment_name,
            initiation_vote_function,
            termination_vote_function,
            env_name,
            num_envs=64,
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
            train_policy_max_steps=1000000,
            policy_success_rate=0.9,
            trial_max_steps=100000,
            max_option_tries=5):
        assert device_type in ["cpu", "cuda"]

        self.num_envs = num_envs
        self.device = torch.device(device_type)
        self.env_name = env_name
        self.trial_max_steps = trial_max_steps
        self.max_option_tries = max_option_tries

        self.option = MujocoOption(
            device=self.device,
            initiation_vote_function=initiation_vote_function,
            termination_vote_function=termination_vote_function
        )

        random.seed(seed)
        self.seed = seed

        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, self.name, str(self.seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)

        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(
            filename=log_file,
            format='%(asctime)s %(levelname)s: %(message)s',
            level=logging.INFO
        )

        logging.info("[experiment] Beginning experiment {} seed {}")

        self.trial_data = {
            "name": [],
            "performance": [],
            "start_pos": [],
            "end_pos": [],
            "true_terminations": [],
            "completed": []
        }

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
                env_name,
                train_policy_max_steps,
                policy_success_rate
            )

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
            logging.warning('[experiment] No negative files were gievn for the termination set.')

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
            env_name,
            max_steps,
            success_rate):
        pass

    def make_vector_env(self, eval=False):
        """vector environment for mujoco"""
        # ant mujoco env
        venv = make_ant_env(self.env_name, self.num_envs, eval=eval)

        return venv

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

    def run_trial(
        self,
        number_episodes_in_trial,
        eval,
        trial_name=""):

        logging.info("[experiment] Starting trial {}".format(trial_name))
        print("[experiment] Starting trial {}".format(trial_name))

        results = []
        start_positions = []
        end_positions = []
        succeeded = []

        start_pos = env.get_position()

        for x in range(number_episodes_in_trial):
            logging.info("Episode {}/{}".format(x, number_episodes_in_trial))

            env = make_ant_env(self.env_name, 1, eval=True)
            steps = [0]
            state = env.reset()
            agent_state = env.render_camera(imshow=False)
            attempt = 0
            timedout = 0
            must_break = False
            infos = [{"position": env.get_position()}]

            while (steps[0]<self.trial_max_steps) and (timedout<3) and (not must_break) and (attempt<self.max_option_tries):

                attempt += 1 
                logging.info("Attempt {}/{}".format(attempt, self.max_option_tries))
                can_initiate = self.option.can_initiate(agent_state, infos[0]["position"])

                if not can_initiate:
                    must_break = True
                else:
                    if eval:
                        state, total_reward, done, infos, steps = self.option.evaluate(
                            env,
                            state,
                            infos,
                            steps,
                            use_global_classifiers=eval
                        )

                    else:
                        state, total_reward, done, infos, steps = self.option.run(
                            env,
                            state,
                            infos,
                            steps,
                            0,
                            0,
                            use_global_classifiers=eval
                        )

                    if done:
                        must_break = True
                        succeeded.append(infos[0]["position"])

                    if infos[0]["option_timed_out"]:
                        timedout += 1
                        logging.info("[experiment] option timed out {} times".format(timedout))

        self.trial_data["name"].append(trial_name)
        self.trial_data["success"].append(infos[0]["success"])
        self.trial_data["start_pos"].append(start_pos)
        self.trial_data["end_pos"].append(infos[0]["position"])

        if not eval:
            self.option.train_initiation(5, 10)
            self.option.train_termination(5, 10)

        logging.info("[experiment] Finished trial {}".format(trial_name))
        print("[experiment] Finished trial {}".format(trial_name))
