from portable.option.memory import SetDataset 
from portable.option.ensemble.custom_attention import *
import matplotlib.pyplot as plt 
from tqdm import tqdm

files = [
    'resources/minigrid_images/adv_doorkey_random_states.npy',
    # 'resources/minigrid_images/adv_doorkey_getredkey_doorred_0_initiation_positive.npy',
    # 'resources/minigrid_images/adv_doorkey_getredkey_doorred_0_initiation_negative.npy',
]

log_dir_base = "runs/advanced_doorkey/encoder"
log_dir = os.path.join(log_dir_base, "0")


dataset = SetDataset(batchsize=128, max_size=300000)
dataset.add_true_files(files)

num_features = 8
feature_size = 200
lr = 1e-3

model = AutoEncoder(3, num_features, feature_size, image_height=128, image_width=128)


device = torch.device("cuda")
model.to(device)

dataset.shuffle()
x, _ = dataset.get_batch()
x = x.to(device)

for i in range(10):
    fig, axes = plt.subplots(ncols=2)
    sample = x[i]
    with torch.no_grad():
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        with torch.no_grad():
            pred = model(sample.unsqueeze(0)).cpu().numpy()
        axes[0].imshow(np.transpose(sample.cpu().numpy(), axes=(1,2,0)))
        axes[1].imshow(np.transpose(pred[0], axes=(1,2,0)))

        fig.savefig(os.path.join(log_dir, "{}.png".format(i)))

    plt.close(fig)


