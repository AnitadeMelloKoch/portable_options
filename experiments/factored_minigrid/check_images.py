import numpy as np 
import matplotlib.pyplot as plt

file_name = "resources/minigrid_images/doorkey_getkey_0_termination_image_positive.npy"


images = np.load(file_name)

fig, axes = plt.subplots(nrows=1, ncols=6)
for image in images:
    for idx, ax in enumerate(axes):
        ax.set_axis_off()
        ax.imshow(image[idx], cmap='gray')
    
    plt.show(block=False)
    input("continue")
plt.close(fig)
