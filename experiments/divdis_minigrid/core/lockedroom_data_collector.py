from experiments.minigrid.utils import environment_builder
import numpy as np
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import LockedRoomPolicyTrainWrapper
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

seed = 0
colours = ["blue", "red", "green", "purple",
           "grey", "yellow"]
data_type = "negative"

for seed in range(2,10):
    env = LockedRoomPolicyTrainWrapper(environment_builder('LockedRoom-v0',
                                seed=seed,
                                grayscale=False,
                                normalize_obs=False,
                                scale_obs=True,
                                final_image_size=(128,128),
                                max_steps=2000))

    for colour in colours:
        states = []
        if data_type is "positive":
            doors_open = [colour]
            doors_closed = []
        else:
            doors_closed = [colour]
            doors_open = []

        for _ in tqdm(range(2000)):
            state, _ = env.reset(random_start=True,
                                random_move_agent=True,
                                random_doors_open=doors_open,
                                random_doors_closed=doors_closed)

            states.append((state/255.0).numpy())

        np.save("resources/large_minigrid_images/lockedroom_opendoor_door{}_{}_termination_{}.npy".format(
            colour, seed, data_type), states)


