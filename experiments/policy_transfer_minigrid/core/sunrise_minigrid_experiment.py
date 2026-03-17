"""Base experiment class for SUNRISE MinGrid experiments."""
import logging
import datetime
import os
import gin
import numpy as np
import torch
from collections import deque
from portable.utils.utils import set_seed
from experiments.experiment_logger import VideoGenerator


@gin.configurable
class SunriseMinigridExperiment:
    """Base experiment class for training SUNRISE agents on MiniGrid environments."""
    
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 policy_phi,
                 use_gpu,
                 make_videos=False):
        """Initialize the SUNRISE experiment.
        
        Args:
            base_dir: Base directory for saving results
            experiment_name: Name of the experiment
            seed: Random seed
            policy_phi: Observation preprocessing function
            use_gpu: GPU device ID (-1 for CPU)
            make_videos: Whether to generate videos
        """
        self.name = experiment_name
        self.seed = seed
        self.use_gpu = use_gpu
        self.policy_phi = policy_phi
        
        # Setup directories
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Set random seed
        set_seed(seed)
        
        # Setup logging
        log_file = os.path.join(self.log_dir, 
                               "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                           format='%(asctime)s %(levelname)s: %(message)s',
                           level=logging.INFO)
        
        # Video generator
        if make_videos:
            self.video_generator = VideoGenerator(os.path.join(self.base_dir, "videos"))
        else:
            self.video_generator = None
        
        self.agent = None
    
    def train_policy(self, 
                    agent,
                    envs,
                    max_steps=500000,
                    eval_interval=10000,
                    eval_episodes=20):
        """Train SUNRISE agent on given environments.
        
        Args:
            agent: SunriseDQNAgent instance
            envs: List of training environments
            max_steps: Maximum training steps
            eval_interval: Steps between evaluations
            eval_episodes: Number of episodes per evaluation
            
        Returns:
            List of (step, success_rate) tuples
        """
        self.agent = agent
        step = 0
        episode = 0
        successes = []
        train_rewards = deque(maxlen=200)
        
        logging.info("Starting SUNRISE training...")
        logging.info(f"Max steps: {max_steps}, Eval interval: {eval_interval}")
        
        while step < max_steps:
            # Sample random environment
            env = np.random.choice(envs)
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = agent.act(obs)
                next_obs, reward, done, info = env.step(action)

                agent.observe(obs, action, reward, next_obs, done)

                obs = next_obs
                episode_reward += reward
                step += 1

            train_rewards.append(episode_reward)
            episode += 1

            # Evaluate periodically (at episode boundaries to avoid corrupting env state)
            if step >= eval_interval * len(successes) + eval_interval:
                eval_success = self.evaluate_policy(agent, envs[0], eval_episodes)
                successes.append((step, eval_success))
                logging.info(f"Step {step}: Success rate = {eval_success:.2%}, "
                           f"Avg train reward = {np.mean(train_rewards):.2f}")

                # Save checkpoint
                agent.save(os.path.join(self.save_dir, f"step_{step}"))

            if episode % 100 == 0:
                logging.info(f"Episode {episode}, Step {step}, "
                           f"Avg reward: {np.mean(train_rewards):.2f}")
        
        # Final save
        agent.save(self.save_dir)
        logging.info("Training complete!")
        
        return successes
    
    def evaluate_policy(self, agent, env, num_episodes=20):
        """Evaluate the agent's success rate.
        
        Args:
            agent: SunriseDQNAgent instance
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes
            
        Returns:
            Success rate (float between 0 and 1)
        """
        agent.training = False
        successes = 0
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)

                if reward > 0:
                    successes += 1
                    break
        
        agent.training = True
        return successes / num_episodes
    
    def save_results(self, successes, filename="success_rates.npy"):
        """Save evaluation results.
        
        Args:
            successes: List of (step, success_rate) tuples
            filename: Name of file to save to
        """
        save_path = os.path.join(self.save_dir, filename)
        np.save(save_path, np.array(successes))
        logging.info(f"Saved results to {save_path}")
