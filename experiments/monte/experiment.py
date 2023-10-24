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
        train_initiation_embedding_epochs=100,
        train_initiation_classifier_epochs=100,
        train_termination=True,
        termination_positive_files=[],
        termination_negative_files=[],
        termination_priority_negative_files=[],
        train_termination_embedding_epochs=100,
        train_termination_classifier_epochs=100,
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
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.env = experiment_env_function(self.seed)

        self._get_percent_completed = get_percentage_function
        self._check_termination_correct = check_termination_true_function
        self.max_option_tries = max_option_tries

        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(
            filename=log_file,
            format='%(asctime)s %(levelname)s: %(message)s',
            level=logging.info
        )

        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        
        self.load()

        if train_initiation is True:
            self._train_initiation(
                initiation_positive_files,
                initiation_negative_files,
                initiation_priority_negative_files,
                train_initiation_embedding_epochs,
                train_initiation_classifier_epochs,
            )
        
        if train_termination is True:
            self._train_termination(
                termination_positive_files,
                termination_negative_files,
                termination_priority_negative_files,
                train_termination_embedding_epochs,
                train_termination_classifier_epochs,
            )

        if train_policy is True:
            self._train_policy(
                policy_bootstrap_env,
                train_policy_max_steps,
                train_policy_success_rate
            )

        self.trial_data = pd.DataFrame([], 
            columns=[
                "normalized_distance",
                "start_position",
                "end_position",
                "true_terminations",
                "completed",
                "dead",
                "steps"], index=pd.Index([], name='name')
        )


    def _train_initiation(
            self,
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

    def _train_termination(
            self,
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
        self.option.save(save_dir)
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
            max_episodes_in_trial,
            eval, 
            trial_name,
            use_agent_space=False):
        assert isinstance(possible_inits, list)
        logger.info("[experiment:run_trial] Starting trial {}".format(trial_name))
        print("[experiment:run_trial] Starting trial {}".format(trial_name))

        results = []
        start_poses = []
        end_poses = []
        true_terminationses = []
        completeds = []
        deads = []
        stepses = []
        instantiation_instances = set()

        episode_count = 0
        instance_well_trained = False

        while episode_count < max_episodes_in_trial and (not instance_well_trained):
            logger.info("Episode {}/{}".format(episode_count, max_episodes_in_trial))
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
                attempt += 1
                logger.info("Attempt {}/{}".format(attempt, self.max_option_tries))

                option_result = self.option.run(
                    self.env,
                    state,
                    info,
                    eval
                )

                if self.option.markov_idx is not None:
                    instantiation_instances.add(self.option.markov_idx)

                if option_result is None:
                    logger.info("[experiment:run_trial] Option did not initiate")
                    result = 0
                    must_break = True
                    position = info["position"]
                    completed = False
                    steps = 0

                else:
                    _, _, done, info, steps = option_result

                    if info["needs_reset"]:
                        logger.info("[experiment:run_trial] Environment needs reset")
                        must_break = True
                    if info["option_timed_out"]:
                        timedout += 1
                        logger.info("[experiment:run_trial] Option timed out ({}/{})".format(timedout, 3))

                    if done:
                        must_break = True

                    agent_state = info["stacked_agent_state"]
                    position = info["position"]

                    completed = self._check_termination_correct(position, true_terminations[rand_idx], self.env)
                    if completed:
                        must_break = True

            episode_count += 1
            instance_well_trained = all([self.option.markov_instantiations[instance].is_well_trained() for instance in instantiation_instances])
            
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
            stepses.append(steps)
            logger.info("Succeeded: {}".format(completed))


        d = pd.DataFrame(
            [
                {
                    "normalized_distance": np.mean(results),
                    "start_position": start_poses,
                    "end_position": end_poses,
                    "true_terminations": true_terminationses,
                    "completed": completeds,
                    "dead": deads,
                    "steps": stepses
                }
            ], index=pd.Index(data=[trial_name], name='name')
        )
            
        self.trial_data = self.trial_data.append(d)

        logger.info("[experiment:run_trial] Finished trial {} performance: {}".format(
            trial_name,
            np.mean(results)
            ))
        logger.info("[experiment:run_trial] All instances well trained: {}".format(
            instance_well_trained))
        if not instance_well_trained:
            logger.info("[experiment:run_trial] instance success rates:"
                .format([self.option.markov_instantiations[instance].is_well_trained() for instance in instantiation_instances])
            )

        print("[experiment] Finished trial {} performance: {}".format(
            trial_name,
            np.mean(results)
            ))
        return instantiation_instances
        
    def test_assimilate(
        self,
        possible_inits,
        true_terminations,
        instantiation_instances,
        max_episodes_in_trial,
        trial_name,
        use_agent_space=False
    ):
        "Run assimilation phase and see if option can be reincorporated into original option"
        assert isinstance(possible_inits, list)

        if isinstance(instantiation_instances, set):
            instantiation_instances = list(instantiation_instances)

        if len(instantiation_instances) == 0:
            logger.info("[experiment:test_assimilation] No instances to test.")

        logger.info("[experiment:test_assimilation] Starting assimilation test.")
        logger.info(instantiation_instances)
        print("[experiment:test_assimilation] Starting assimilation test.")

        results = []
        start_poses = []
        end_poses = []
        true_terminationses = []
        completeds = []
        deads = []
        stepses = []

        episode_count = 0
        instance_succeeded = False

        for instance_idx in instantiation_instances:
            instance = self.option.markov_instantiations[instance_idx]
            while episode_count < max_episodes_in_trial and (not instance_succeeded):
                logger.info("Assimilation test {}/{}".format(episode_count, max_episodes_in_trial))
                completed = False
                episode_count += 1

                rand_idx = random.randint(0, len(possible_inits) - 1)
                rand_state = possible_inits[rand_idx]
                state = self._set_env_ram(
                    rand_state["ram"],
                    rand_state["state"],
                    rand_state["agent_state"],
                    use_agent_space
                )
                termination = true_terminations[rand_idx]
                
                agent_state = rand_state["agent_state"]
                start_pos = rand_state["position"]
                info = self.env.get_current_info({})

                completed = False
                position = info["position"]
                dead = False
                result = 0
                steps = 0

                if instance.can_initiate((info["player_x"], info["player_y"])):
                    _, _, done, info, steps = instance.assimilate_run(
                        self.env,
                        state,
                        info
                    )

                    position = info["position"]
                    dead = info["dead"]
                    completed = self._check_termination_correct(
                        position,
                        termination,
                        self.env
                    )

                    if not dead:
                        result = self._get_percent_completed(
                            start_pos,
                            position,
                            termination,
                            self.env
                        )
                    
                results.append(result)
                start_poses.append(start_pos)
                end_poses.append(position)
                true_terminationses.append(termination)
                completeds.append(completed)
                deads.append(dead)
                stepses.append(steps)
                instance_succeeded = instance.can_assimilate()
                if instance_succeeded is None:
                    instance_succeeded = False
                if instance_succeeded is True:
                    self.option.update_option(
                        instance,
                        200,
                        20,
                        80
                    )
            logger.info("Instance {} succeeded: {} average performance: {}"
                .format(instance_idx, completed, np.mean(results)))
            
            d = pd.DataFrame(
                [
                    {
                        "normalized_distance": np.mean(results),
                        "start_position": start_poses,
                        "end_position": end_poses,
                        "true_terminations": true_terminationses,
                        "completed": completeds,
                        "dead": deads,
                        "steps": stepses
                    }
                ], index=pd.Index(data=[trial_name+str(instance_idx)], name='name')
            )

            self.trial_data = self.trial_data.append(d)

        logger.info("[experiment] Finished trial {} performance: {}".format(
            trial_name,
            np.mean(results)
        ))
        print("[experiment] Finished trial {} performance: {}".format(
            trial_name,
            np.mean(results)
        ))
        self.option.markov_instantiations = []

    def bootstrap_from_room(
            self,
            possible_inits,
            true_terminations, 
            number_episodes_in_trial,
            use_agent_space=False):
        
        assert isinstance(possible_inits, list)
        assert isinstance(true_terminations, list)
        
        logger.info("Bootstrapping weights from training room")
        print("Bootstrapping weights from training room")

        for x in range(number_episodes_in_trial):
            logger.info("Episode {}/{}".format(x, number_episodes_in_trial))
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

            while (not done) and (not info["needs_reset"]) and (count < 100) and (timedout < 3):

                count += 1

                option_result = self.option.run(
                    self.env,
                    state,
                    info,
                    eval=True
                )

                if option_result is None:
                    logger.info("initiation was not triggered")
                    self.option.initiation_update_confidence(was_successful=False, votes=self.option.initiation.votes)
                    break

                _, _, done, info, _ = option_result

                if info['needs_reset']:
                    logger.info("Breaking because environment needs reset")
                    break

                if info['option_timed_out']:
                    logger.info('[experiment] option has timed out {} times'.format(timedout))
                    timedout += 1

                if self._check_termination_correct(self.env.get_current_position(), true_terminations[rand_idx], self.env):
                    self.option.termination_update_confidence(was_successful=True, votes=self.option.termination.votes)
                    logger.info("Breaking because correct termination was found")
                    break
                else:
                    self.option.termination_update_confidence(was_successful=False, votes=self.option.termination.votes)

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

    def test_classifiers(self, possible_inits_true, possible_inits_false):
        assert isinstance(possible_inits_true, list)
        assert isinstance(possible_inits_false, list)


        for idx, instance in enumerate(possible_inits_true):
            state = instance["agent_state"]
            plot_name = os.path.join(self.plot_dir, "attentions_initiation_true")
            os.makedirs(plot_name, exist_ok=True)
            plot_state(state, plot_name+'/'+str(idx)+'.png')

            print("Instance {}/{}".format(idx, len(possible_inits_true)))

            votes, class_conf, confidences, attentions = self.option.initiation.get_attentions(state)
            plot_dir = os.path.join(self.plot_dir, "attentions_initiation_true", "initiation", str(idx))
            os.makedirs(plot_dir, exist_ok=True)

            plot_attentions(attentions, votes, class_conf, confidences, plot_dir)

            votes, class_conf, confidences, attentions = self.option.termination.get_attentions(state)
            plot_dir = os.path.join(self.plot_dir, "attentions_initiation_true", "termination", str(idx))
            os.makedirs(plot_dir, exist_ok=True)

            plot_attentions(attentions, votes, class_conf, confidences, plot_dir)

        for idx, instance in enumerate(possible_inits_false):
            state = self._set_env_ram(
                instance["ram"],
                instance["state"],
                instance["agent_state"],
                True
            )
            plot_name = os.path.join(self.plot_dir, "attentions_initiation_false")
            os.makedirs(plot_name, exist_ok=True)
            plot_state(state, plot_name+'/'+str(idx)+'.png')

            print("Instance {}/{}".format(idx, len(possible_inits_false)))
            votes, class_conf, confidences, attentions = self.option.initiation.get_attentions(state)
            plot_dir = os.path.join(self.plot_dir, "attentions_initiation_false", "initiation", str(idx))
            os.makedirs(plot_dir, exist_ok=True)

            plot_attentions(attentions, votes, class_conf, confidences, plot_dir)

            votes, class_conf, confidences, attentions = self.option.termination.get_attentions(state)
            plot_dir = os.path.join(self.plot_dir, "attentions_initiation_false", "termination", str(idx))
            os.makedirs(plot_dir, exist_ok=True)

            plot_attentions(attentions, votes, class_conf, confidences, plot_dir)





