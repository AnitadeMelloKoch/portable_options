import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocess import Pool
from functools import partial

from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import \
    AdvancedDoorKeyPolicyTrainWrapper
from experiments.minigrid.utils import (environment_builder,
                                        factored_environment_builder)


seed = 11


    def collect_seed(seed, initiation=None, termination=None):
        colours = ["red", "green", "blue", "purple", "yellow", "grey"]
        
        obs, _ = env.reset(random_start=True,
                        keep_colour=colour,
                        agent_reposition_attempts=repos_attempts)
        
        states.append(obs.numpy())


            states = np.array(states)

            if (initiation is True) or (termination is False): # init pos, term neg
                np.save("resources/factored_minigrid_images/adv_doorkey_8x8_v2_get{}key_door{}_{}_1_initiation_positive.npy".format(colour, colour, seed), states)
                np.save("resources/factored_minigrid_images/adv_doorkey_8x8_v2_get{}key_door{}_{}_1_termination_negative.npy".format(colour, colour, seed), states)
            elif (initiation is False) or (termination is True): # init neg, term pos
                np.save("resources/factored_minigrid_images/adv_doorkey_8x8_v2_get{}key_door{}_{}_1_initiation_negative.npy".format(colour, colour, seed), states)
                np.save("resources/factored_minigrid_images/adv_doorkey_8x8_v2_get{}key_door{}_{}_1_termination_positive.npy".format(colour, colour, seed), states)
            else:
                raise ValueError("initiation and termination cannot be the same")

            print(f"Saved seed {seed}, colour {colour}, initiation {initiation}, termination {termination}")
            print(f"states shape: {states.shape}")

    np.save("resources/factored_minigrid_images/adv_doorkey_8x8_v2_get{}key_door{}_{}_1_termination_negative.npy".format(colour, colour, seed)
            , states)


    # multiprocessing
    seeds = [0,1,2,3,4,5,6,7,8,9,10,11]

    with Pool() as p:
        p.map(partial(collect_seed, termination=True), seeds)


    with Pool() as p:
        p.map(partial(collect_seed, termination=False), seeds)

    



