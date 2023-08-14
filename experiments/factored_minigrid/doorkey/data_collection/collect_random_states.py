from experiments.factored_minigrid.doorkey.core.policy_train_wrapper import FactoredDoorKeyPolicyTrainWrapper
import numpy as np 
import matplotlib.pyplot as plt
from experiments.factored_minigrid.utils import environment_builder
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

fig, axes = plt.subplots(ncols=6)

for x in range(1):
    for y in tqdm(range(5000)):
        seed = np.random.randint(low=3, high=20000)
        env = FactoredDoorKeyPolicyTrainWrapper(
            environment_builder('FactoredMiniGrid-DoorKey-8x8-v0',
                                seed=seed),
            lambda x: False,
            key_picked_up=key[x],
            door_open=door[x]
        )
        
        # print("key:", key[x])
        # print("door:", door[x])
        
        obs, _ = env.reset()
        
        collections[x].append(obs)
                
        # for idx in range(6):
        #     axes[idx].set_axis_off()
        #     axes[idx].imshow(obs[idx])
        
        # plt.show(block=False)
        # input("continue")

plt.close(fig)

print(len(collections[0]))
print(len(collections[1]))
print(len(collections[2]))

np.save("resources/minigrid_images/doorkey_start_2.npy", collections[0])
# np.save("resources/minigrid_images/doorkey_no_key.npy", collections[1])
# np.save("resources/minigrid_images/doorkey_open_door.npy", collections[2])


