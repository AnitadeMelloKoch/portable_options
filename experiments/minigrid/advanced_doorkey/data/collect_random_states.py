import random
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import \
    AdvancedDoorKeyPolicyTrainWrapper
from experiments.minigrid.utils import environment_builder, factored_environment_builder


def collect_seed(seed, task, initiation=None, termination=None):
    colours = ["red", "green", "blue", "purple", "yellow", "grey"]        
    for colour in colours:
        states = []
        for _ in tqdm(range(5000)):
            repos_attempts = np.random.randint(low=0, high=1000)
            env = environment_builder('AdvancedDoorKey-16x16-v0', seed=seed, grayscale=False,
                                      scale_obs=True, final_image_size=(84,84), normalize_obs=False)
            #env = factored_environment_builder('AdvancedDoorKey-8x8-v0', seed=seed)
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import \
    AdvancedDoorKeyPolicyTrainWrapper
from experiments.minigrid.utils import environment_builder, factored_environment_builder


def collect_seed(seed, task, initiation=None, termination=None):
    colours = ["red", "green", "blue", "purple", "yellow", "grey"]        
    for colour in colours:
        states = []
        for _ in tqdm(range(5000)):
            repos_attempts = np.random.randint(low=0, high=1000)
            env = environment_builder('AdvancedDoorKey-16x16-v0', seed=seed, grayscale=False,
                                      scale_obs=True, final_image_size=(84,84), normalize_obs=False)
            #env = factored_environment_builder('AdvancedDoorKey-8x8-v0', seed=seed)
            env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                                    door_colour=colour,
                                                    image_input=True)
            
            if task == "get_key":
                if (initiation is True) or (termination is False): # key presnet; can get key: init pos, term neg
                    obs, _ = env.reset(random_start=True, 
                                    keep_colour=colour,
                                    agent_reposition_attempts=repos_attempts)
                elif (initiation is False) or (termination is True): # key not presnet; cant get key:init neg, term pos
                    obs, _ = env.reset(random_start=True, 
                                    pickup_colour=colour,
                                    agent_reposition_attempts=repos_attempts)


            elif task == "open_door":
                if (initiation is True) or (termination is False): # door closed; can open door: init pos, term neg
                    obs, _ = env.reset(random_start=True, 
                                    force_door_closed=True,
                                    agent_reposition_attempts=repos_attempts)
                elif (initiation is False) or (termination is True): # door open; cant open door: init neg, term pos
                    obs, _ = env.reset(random_start=True, 
                                    force_door_open=True,
                                    agent_reposition_attempts=repos_attempts)
            else:
                raise ValueError("task must be either 'get_key' or 'open_door'")
                
                
            states.append(obs.numpy())

        states = np.array(states)
        base_dir = "resources/minigrid_images/" 
        task_name = f"get{colour}key" if task == "get_key" else f"open{colour}door"
        if (initiation is True) or (termination is False): # init pos, term neg
            np.save(base_dir+f"adv_doorkey_16x16_v2_{task_name}_door{colour}_{seed}_1_initiation_positive.npy", states)
            np.save(base_dir+f"adv_doorkey_16x16_v2_{task_name}_door{colour}_{seed}_1_termination_negative.npy", states)
        elif (initiation is False) or (termination is True): # init neg, term pos
            np.save(base_dir+f"adv_doorkey_16x16_v2_{task_name}_door{colour}_{seed}_1_initiation_negative.npy", states)
            np.save(base_dir+f"adv_doorkey_16x16_v2_{task_name}_door{colour}_{seed}_1_termination_positive.npy", states)
        else:
            raise ValueError("initiation and termination cannot be the same")

        print(f"Saved seed {seed}, colour {colour}, initiation {initiation}, termination {termination}")
        print(f"states shape: {states.shape}")



if __name__ == '__main__':

    # multiprocessing
    task = 'open_door'
    seeds = [0,1,2,3,4,5,6,7,8,9,10,11]
    USE_MP = True
    
    if USE_MP:
        import multiprocess as mp
        
        with mp.Pool() as p:
            p.map(partial(collect_seed, task=task, termination=True), seeds)

        with mp.Pool() as p:
            p.map(partial(collect_seed, task=task, termination=False), seeds)
        
    else:
        for seed in tqdm(seeds):
            collect_seed(seed, task=task, termination=True)
            collect_seed(seed, task=task, termination=False)