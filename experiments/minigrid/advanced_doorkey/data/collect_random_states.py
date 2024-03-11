import random
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from multiprocess import Pool
from tqdm import tqdm

from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import \
    AdvancedDoorKeyPolicyTrainWrapper
from experiments.minigrid.utils import (environment_builder,
                                        factored_environment_builder)

if __name__ == '__main__':

    def collect_seed(seed, initiation=None, termination=None):
        colours = ["red", "green", "blue", "purple", "yellow", "grey"]
        
        for colour in colours:
            states = []
            for _ in range(2000):
                repos_attempts = np.random.randint(low=0, high=1000)
                env = environment_builder('AdvancedDoorKey-8x8-v0', seed=seed, grayscale=False)
                #env = factored_environment_builder('AdvancedDoorKey-8x8-v0', seed=seed)
                env = AdvancedDoorKeyPolicyTrainWrapper(env,
                                                        door_colour=colour,
                                                        image_input=True)
                
                if (initiation is True) or (termination is False): # init pos, term neg
                    obs, _ = env.reset(random_start=True, 
                                       keep_colour=colour,
                                       agent_reposition_attempts=repos_attempts)
                elif (initiation is False) or (termination is True): # init neg, term pos
                    obs, _ = env.reset(random_start=True, 
                                       pickup_colour=colour,
                                       agent_reposition_attempts=repos_attempts)
                else:
                    raise ValueError("initiation and termination cannot be the same")
                
                states.append(obs.numpy())

            states = np.array(states)
            base_dir = "resources/minigrid_images/" 
            if (initiation is True) or (termination is False): # init pos, term neg
                np.save(base_dir+"adv_doorkey_8x8_v2_get{}key_door{}_{}_1_initiation_positive.npy".format(colour, colour, seed), states)
                np.save(base_dir+"adv_doorkey_8x8_v2_get{}key_door{}_{}_1_termination_negative.npy".format(colour, colour, seed), states)
            elif (initiation is False) or (termination is True): # init neg, term pos
                np.save(base_dir+"adv_doorkey_8x8_v2_get{}key_door{}_{}_1_initiation_negative.npy".format(colour, colour, seed), states)
                np.save(base_dir+"adv_doorkey_8x8_v2_get{}key_door{}_{}_1_termination_positive.npy".format(colour, colour, seed), states)
            else:
                raise ValueError("initiation and termination cannot be the same")

            print(f"Saved seed {seed}, colour {colour}, initiation {initiation}, termination {termination}")
            print(f"states shape: {states.shape}")


    # multiprocessing
    seeds = [0,1,2,3,4,5,6,7,8,9,10,11]

    with Pool() as p:
        p.map(partial(collect_seed, termination=True), seeds)

    with Pool() as p:
        p.map(partial(collect_seed, termination=False), seeds)

    



