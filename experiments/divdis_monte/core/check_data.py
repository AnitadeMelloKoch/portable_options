from portable.option.memory import SetDataset
import matplotlib.pyplot as plt 
import numpy as np

files = [
    'resources/monte_images/lasers2_toleft_room7_termination_positive.npy',
]

dataset = SetDataset(batchsize=1)

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot()

dataset.add_true_files(files)
dataset.set_transform_function(lambda x: x)
print(dataset.true_length)

gs = fig.add_gridspec(nrows=1, ncols=4)
data = []
count = 0
for _ in range(dataset.num_batches):
    x, _ = dataset.get_batch()
    for im in x:
        count += 1
        print(count)
        for idx in range(4):
            ax = fig.add_subplot(gs[0,idx])
            ax.imshow(im[idx], cmap='gray')
            ax.axis('off')
        plt.show(block=False)
        keep = input("press y to keep")
        if keep == 'y':
            data.append(x.numpy())
            

data = np.array(data)

print(data[0])
print(len(data))

np.save(files[0], data)