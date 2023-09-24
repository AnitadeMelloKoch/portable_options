from procgen import ProcgenEnv
import gin
from experiments.procgen.core.vec_env import VecExtractDictObs, VecNormalize, VecChannelOrder, VecMonitor
import pandas as pd 
from portable.option.policy.agents.batch_ensemble_agent import BatchedEnsembleAgent
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import datetime
from portable.utils.utils import set_seed
from portable.option.ensemble.custom_attention import AutoEncoder
import lzma
import dill
import torch
import matplotlib.pyplot as plt 
import numpy as np
import random
from collections import deque
from portable.option.policy.agents import evaluating 

@gin.configurable
class ProcgenExperiment():
    def __init__(self,
                 use_gpu,
                 base_dir,
                 experiment_name,
                 experiment_seed,
                 policy_phi,
                 env_name,
                 batch_size,
                 warmup_steps,
                 prioritized_replay_anneal_steps,
                 buffer_length,
                 update_interval,
                 q_target_update_interval,
                 final_epsilon,
                 final_exploration_frames,
                 ppo_lambda,
                 value_function_coef,
                 entropy_coef,
                 clip_range,
                 max_grad_norm,
                 embedding_phi,
                 policy_epochs=10,
                 attention_num=3,
                 learning_rate=5e-4,
                 discount_rate=0.9,
                 c=500,
                 gru_hidden_size=128,
                 divergence_loss_scale=1,
                 num_levels=20,
                 num_envs=5,
                 evaluate_num_levels=5,
                 distribution_mode="hard",
                 max_steps_per_level=1e6):
        
        
        self.warmup_steps = warmup_steps
        self.prioritized_replay_anneal_steps = prioritized_replay_anneal_steps
        self.buffer_length = buffer_length
        self.update_interval = update_interval
        self.q_target_update_interval = q_target_update_interval
        self.final_epsilon = final_epsilon
        self.final_exploration_frames = final_exploration_frames
        self.batch_size = batch_size
        self.embedding_phi = embedding_phi
        
        self.base_dir = os.path.join(base_dir, experiment_name)
        self.use_gpu = use_gpu
        self.experiment_name = experiment_name
        self.experiment_seed = experiment_seed
        set_seed(experiment_seed)
        self.attention_num = attention_num
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.c = c
        self.gru_hidden_size = gru_hidden_size
        self.divergence_loss_scale = divergence_loss_scale
        self.max_steps_per_level = max_steps_per_level
        self.evaluate_num_levels = evaluate_num_levels
        self.ppo_lambda = ppo_lambda
        self.value_function_coef = value_function_coef
        self.entropy_coef = entropy_coef
        self.policy_epochs = policy_epochs
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        
        if self.use_gpu:
            self.embedding = AutoEncoder().to("cuda")
        else:
            self.embedding = AutoEncoder()
        self.embedding_loaded = False
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.experiment_name, self.experiment_seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Experiment seed: {}".format(experiment_seed))
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.policy_phi = policy_phi
        
        self.num_levels = num_levels
        self.env_name = env_name
        self.num_envs = num_envs
        self.distribution_mode = distribution_mode
        
        self.trial_data_train = []
        self.trial_data_eval = []
        
        self.agent = None

    def save(self):
        self.agent.save(os.path.join(self.save_dir, 'policy'))
        filename = os.path.join(self.save_dir, 'experiment_data_train.pkl')
        with lzma.open(filename, 'wb') as f:
            dill.dump(self.trial_data_train, f)
        filename = os.path.join(self.save_dir, 'experiment_data_eval.pkl')
        with lzma.open(filename, 'wb') as f:
            dill.dump(self.trial_data_eval, f)
    
    def load(self):
        self.agent.load(os.path.join(self.save_dir, 'policy'))
        filename = os.path.join(self.save_dir, 'experiment_data_train.pkl')
        if os.path.exists(filename):
            with lzma.open(filename, 'rb') as f:
                self.trial_data_train = dill.load(f)
        filename = os.path.join(self.save_dir, 'experiment_data_eval.pkl')
        if os.path.exists(filename):
            with lzma.open(filename, 'rb') as f:
                self.trial_data_eval = dill.load(f)
    
    def _make_agent(self, env):
        self.agent = BatchedEnsembleAgent(embedding=self.embedding,
                                          embedding_phi=self.embedding_phi,
                                          learning_rate=self.learning_rate,
                                          num_modules=self.attention_num,
                                          use_gpu=self.use_gpu,
                                          warmup_steps=self.warmup_steps,
                                          batch_size=self.batch_size,
                                          phi=self.policy_phi,
                                          buffer_length=self.buffer_length,
                                          update_interval=self.update_interval,
                                          discount_rate=self.discount_rate,
                                          bandit_exploration_weight=self.c,
                                          num_actions=env.action_space.n,
                                          ppo_lambda=self.ppo_lambda,
                                          value_function_coef=self.value_function_coef,
                                          entropy_coef=self.entropy_coef,
                                          num_envs=self.num_envs,
                                          step_epochs=self.policy_epochs,
                                          clip_range=self.clip_range,
                                          max_grad_norm=self.max_grad_norm,
                                          divergence_loss_scale=self.divergence_loss_scale)
    
    def load_embedding(self, load_dir=None):
        if load_dir is None:
            load_dir = os.path.join(self.save_dir, 'embedding', 'model.ckpt')
        logging.info("[expperiment embedding] Embedding loaded from {}".format(load_dir))
        self.embedding.load_state_dict(torch.load(load_dir))
        self.embedding_loaded = True
    
    def train_embedding(self,
                        train_data,
                        epochs,
                        lr):
        optimizer = torch.optim.Adam(self.embedding.parameters(), lr=lr)
        mse_loss = torch.nn.MSELoss()
        base_dir = os.path.join(self.save_dir, 'embedding')
        os.makedirs(base_dir, exist_ok=True)
        save_file = os.path.join(self.save_dir, 'embedding', 'model.ckpt')
        train_x = None
        
        for epoch in range(epochs):
            train_data.shuffle()
            loss = 0
            counter_train = 0
            logging.info("[experiment embedding] Training embedding")
            for b_idx in range(train_data.num_batches):
                counter_train += 1
                x, _ = train_data.get_batch()
                x = x.to(self.device)
                train_x = x[:5]
                pred = self.embedding(x)
                mse_loss = mse_loss(pred, x)
                loss += mse_loss.item()
                mse_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            self.writer.add_scalar('embedding/mse_loss', loss/counter_train, epoch)
            for i in range(5):
                fig, axes = plt.subplots(ncols=2)
                sample = train_x[i]
                with torch.no_grad():
                    axes[0].set_axis_off()
                    axes[1].set_axis_off()
                    with torch.no_grad():
                        pred = self.embedding(sample.unsqueeze(0)).cpu().numpy()
                    axes[0].imshow(np.transpose(sample.cpu().numpy(), axes=(1,2,0)))
                    axes[1].imshow(np.transpose(pred[0], axes=(1,2,0)))

                    fig.savefig(os.path.join(base_dir, "{}.png".format(i)))
                plt.close(fig)
        torch.save(self.embedding.state_dict(), save_file)
        self.embedding_loaded = True
    
    def make_vector_env(self,
                        level_index,
                        eval=False):
        """
        make a procgen environment such that the training env only focus on 1 particular level
        the testing env will sample randomly from a number of levels

        NOTE: Warning: the eval environment does not guarantee sequential transition between `num_levels` levels.
        The following level is sampled randomly, so evaluation is not deterministic, and performed on god knows
        which levels.
        """
        
        venv = ProcgenEnv(num_envs=self.num_envs,
                          env_name=self.env_name,
                          num_levels=self.evaluate_num_levels if eval else 1,
                          start_level=0 if eval else level_index,
                          distribution_mode=self.distribution_mode,
                          num_threads=4,
                          center_agent=True,
                          rand_seed=self.experiment_seed)
        
        venv = VecChannelOrder(venv, channel_order="chw")
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv, ob=False)
        
        return venv
    
    def run(self):
        train_env = self.make_vector_env(level_index=0, eval=False)
        test_env = self.make_vector_env(level_index=0, eval=True)
        
        self._make_agent(train_env)
        
        train_level_seeds = random.sample(range(10, 100), self.num_levels)
        
        for seed in train_level_seeds:
            train_env = self.make_vector_env(level_index=seed, eval=False)
            self.run_level(train_env,
                           test_env, 
                           seed)
    
    def run_level(self,
                  train_env,
                  test_env,
                  seed):
        logging.info("Begin Training on Level with Seed: {}".format(seed))
        print("Begin Training on Level with Seed: {}".format(seed))
        
        train_obs = train_env.reset()
        train_steps = np.zeros(self.num_envs, dtype=int)
        
        test_obs = test_env.reset()
        test_steps = np.zeros(self.num_envs)
        
        step_cnt = 0
        cumulative_reward_train = 0
        cumulative_reward_test = 0
        
        while step_cnt < self.max_steps_per_level:
            train_obs, train_steps, train_epinfo = self.step(train_env,
                                                             train_obs,
                                                             train_steps)
            self.trial_data_train.extend(train_epinfo)
            cumulative_reward_train += np.mean([train_epinfo[x]["reward"] for x in range(len(train_epinfo))])
            
            with evaluating(self.agent):
                test_obs, test_steps, test_epinfo = self.step(test_env,
                                                              test_obs,
                                                              test_steps)
                self.trial_data_eval.extend(test_epinfo)
                cumulative_reward_test += np.mean([test_epinfo[x]["reward"] for x in range(len(test_epinfo))])
            
            step_cnt += 1
            
            if step_cnt%100 == 0:
                
                print("[Train] Step count: {} Ave reward: {}".format(step_cnt, cumulative_reward_train))
                print("[Eval] Step count: {} Ave reward: {}".format(step_cnt, cumulative_reward_train))
                
                logging.info("[Train] Step count: {} Ave reward: {}".format(step_cnt, cumulative_reward_test))
                logging.info("[Eval] Step count: {} Ave reward: {}".format(step_cnt, cumulative_reward_test))
        
        self.save()
    
    def safe_mean(xs):
        return np.nan if len(xs) == 0 else np.mean(xs)
    
    def step(self, env, obs, steps, env_max_steps=1000):
        action = self.agent.batch_act(obs)
        new_obs, reward, done, infos = env.step(action)
        steps += 1
        reset = (steps == env_max_steps)
        steps[done] = 0
        
        self.agent.batch_observe(batch_obs=new_obs,
                                 batch_reward=reward,
                                 batch_done=done,
                                 batch_reset=reset)
        
        
        epinfo = []
        for idx, info in enumerate(infos):
            info["reward"] = reward[idx]
            info["steps"] = steps[idx]
            
            epinfo.append(info)
        
        return new_obs, steps, epinfo
    






