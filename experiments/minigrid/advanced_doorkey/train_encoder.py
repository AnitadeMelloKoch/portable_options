from portable.option.memory import SetDataset 
from portable.option.ensemble.custom_attention import *
import matplotlib.pyplot as plt 
from tqdm import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter
import os 

files = [
    'resources/minigrid_images/adv_doorkey_random_states.npy',
    'resources/minigrid_images/lockedroom_random_states.npy',
]

def dataset_transform(x):
    b, c, x_size, y_size = x.shape
    pad_x = (152 - x_size)
    pad_y = (152 - y_size)
    
    if pad_x%2 == 0:
        padding_x = (pad_x//2, pad_x//2)
    else:
        padding_x = (pad_x//2 + 1, pad_x//2)
    
    if pad_y%2 == 0:
        padding_y = (pad_y//2, pad_y//2)
    else:
        padding_y = (pad_y//2 + 1, pad_y//2)
    
    transformed_x = np.zeros((b,c,152,152))
    
    for im_idx in range(b):
        transformed_x[im_idx,:,:,:] = np.stack([
            np.pad(x[im_idx,idx,:,:], 
                   (padding_x, padding_y), 
                   mode="constant", constant_values=0) for idx in range(c)
        ], axis=0)
    
    return transformed_x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--feature_size", type=int)
    # args = parser.parse_args()
    
    log_dir_base = "runs/advanced_doorkey/encoder"
    log_dir = os.path.join(log_dir_base, "lockedroom-{}".format(0))
    x = 0
    while os.path.exists(log_dir):
        x += 1
        log_dir = os.path.join(log_dir_base, "lockedroom-{}".format(x))
    
    writer = SummaryWriter(log_dir=log_dir)
    dataset = SetDataset(batchsize=256, max_size=500000, pad_func=dataset_transform)
    dataset.add_true_files(files)
    
    # feature_size = args.feature_size
    feature_size = 500
    lr = 1e-4
    
    model = AutoEncoder(3, feature_size, image_height=152, image_width=152)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    device = torch.device("cuda")
    model.to(device)
    
    train_x = None 
    
    for epoch in range(80):
        dataset.shuffle()
        loss = 0
        mse_losses = 0
        counter_train = 0
        for b_idx in tqdm(range(dataset.num_batches)):
            counter_train += 1
            x, _ = dataset.get_batch()
            x = x.to(device)
            train_x = x
            pred = model(x)
            mse_loss = mse(pred, x)
            mse_losses += mse_loss.item()
            b_loss = mse_loss
            b_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss += b_loss.item()

        writer.add_scalar('train/mse_loss', mse_losses/counter_train, epoch)
        writer.add_scalar('train/total_loss', loss/counter_train, epoch)
        print("Epoch {} mse loss {} total loss {}".format(epoch, 
                                                        mse_losses/counter_train,
                                                        loss/counter_train,
                                                        ))

        torch.save(model.state_dict(), os.path.join(log_dir, "encoder.ckpt"))

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

