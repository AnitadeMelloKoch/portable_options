import torch_ac
from experiments.minigrid.doorkey.core.models.ac_model import ACModel
from torch_ac.utils import ParallelEnv
import time
import gin
from experiments.minigrid.utils import process_data
import tensorboardX

@gin.configurable
class MinigridAgentWrapper():
    """Wraps torch_ac base agent"""
    def __init__(self,
                 save_dir,
                 device,
                 training_env,
                 preprocess_obs,
                 algorithm="ppo",
                 log_interval=1,
                 observation_shape=(128,128,3),
                 action_space=7,
                 gamma=0.99,
                 learning_rate=0.001,
                 gae_lambda=0.95,
                 entropy_coeff=0.01,
                 value_loss_coeff=0.5,
                 max_grad_norm=0.5,
                 optimizer_alpha=0.99,
                 optimizer_eps=1e-8,
                 clip_eps=0.2,
                 epochs=4,
                 batchsize=256):
        
        self.save_dir = save_dir
        self.log_interval = log_interval
        
        self.tb_writer = tensorboardX.SummaryWriter(self.save_dir)
        
        ac_model = ACModel(observation_shape, action_space)
        ac_model.to(device)
        
        self.frames = 0
        
        if algorithm == "a2c":
            self.agent = torch_ac.A2CAlgo([training_env],
                                           ac_model,
                                           device,
                                           5,
                                           gamma,
                                           learning_rate,
                                           gae_lambda,
                                           entropy_coeff,
                                           value_loss_coeff,
                                           max_grad_norm,
                                           1,
                                           optimizer_alpha,
                                           optimizer_eps,
                                           preprocess_obs)
        elif algorithm == "ppo":
            self.agent = torch_ac.PPOAlgo([training_env],
                                           ac_model,
                                           device,
                                           128,
                                           gamma,
                                           learning_rate,
                                           gae_lambda,
                                           entropy_coeff,
                                           value_loss_coeff,
                                           max_grad_norm,
                                           1,
                                           optimizer_eps,
                                           clip_eps,
                                           epochs,
                                           batchsize,
                                           preprocess_obs)
            
    def train(self, envs, num_frames):
        
        self.agent.env = ParallelEnv(envs)
        
        train_frames = 0
        update = 0
        start_time = time.time()
        
        while train_frames < num_frames:
            update_start_time = time.time()
            exps, logs1 = self.agent.collect_experiences()
            logs2 = self.agent.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()
            
            self.frames += logs["num_frames"]
            train_frames += logs["num_frames"]
            update += 1
            
            if update % self.log_interval == 0:
                fps = logs["num_frames"]/(update_end_time-update_start_time)
                duration = int(time.time()-start_time)
                return_per_episode = process_data(logs["return_per_episode"])
                reshaped_return_per_episode = process_data(logs["reshaped_return_per_episode"])
                num_frames_per_episode = process_data(logs["num_frames_per_episode"])
                
                header = ["update", "frames", "FPS", "duration"]
                data = [update, self.frames, fps, duration]
                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()
                header += ["reshaped_return_" + key for key in reshaped_return_per_episode.keys()]
                data += reshaped_return_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
        
                for field, value in zip(header, data):
                    self.tb_writer.add_scalar(field, value, self.frames)