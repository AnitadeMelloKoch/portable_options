from experiments.minigrid.doorkey.core.policy_train_wrapper import DoorKeyPolicyTrainWrapper
import numpy as np 
import matplotlib.pyplot as plt 
from experiments.minigrid.utils import environment_builder
from tqdm import tqdm 

collections = [
    [],
    [],
    []
]

key = [
    False,
    True,
    True
]

door = [
    False,
    False,
    True
]

fig, ax = plt.subplots()

for x in range(3):
    # for y in range(5):
    for y in tqdm(range(10000)):
        seed = np.random.randint(low=3, high=20000)
        # env = DoorKeyPolicyTrainWrapper(
        #     # environment_builder('MiniGrid-DoorKey-8x8-v0',
        #     #                     seed=seed,
        #     #                     grayscale=False),
        #     environment_builder('MiniGrid-DoorKey-8x8-v0',
        #                         seed=seed,
        #                         grayscale=False,
        #                         scale_obs=True),
        #     lambda x: False,
        #     key_picked_up=key[x],
        #     door_open=door[x]
        # )
        
        env = environment_builder('MiniGrid-DoorKey-16x16-v0',
                                seed=seed,
                                grayscale=False,
                                pad_obs=False)
        
        # print("key:", key[x])
        # print("door:", door[x])
        
        obs, _ = env.reset()
        
        obs, _, _, _ = env.step(1)
        
        collections[x].append(obs)
        
        print(obs.shape)
        
        ax.set_axis_off()
        ax.imshow(np.transpose(obs, axes=(1,2,0)))
        
        plt.show(block=False)
        input("continue")

plt.close(fig)

# print(len(collections[0]))
# print(len(collections[1]))
# print(len(collections[2]))

# np.save("resources/minigrid_images/doorkey_colour_start.npy", collections[0])
# np.save("resources/minigrid_images/doorkey_colour_no_key.npy", collections[1])
# np.save("resources/minigrid_images/doorkey_colour_open_door.npy", collections[2])