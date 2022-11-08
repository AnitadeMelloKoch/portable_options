import numpy as np
import glob
import matplotlib.pyplot as plt

path = "/home/anita/code/portable_options/resources/classifier_data/goal/termination/positive"

def concat(arr1, arr2):
    if len(arr1) == 0:
        return arr2
    else:
        return np.concatenate((arr1, arr2), axis=0)

data = np.array([])



for name in glob.glob(path + "/*"):
    # fig = plt.figure(num=1, clear=True)
    # ax = fig.add_subplot()
    # data = np.load(name)
    # print(name)
    # ax.imshow(data)
    # plt.show(block=False)
    # plt.pause(0.1)
    new_data = np.expand_dims(np.load(name), axis=0)
    data = concat(data, new_data)

data = np.reshape(data, (-1, 3, 128, 128))
print(data.shape)

save_file = '/home/anita/code/portable_options/resources/mujoco_images/goal_termination_positive.npy'
np.save(save_file, data)
