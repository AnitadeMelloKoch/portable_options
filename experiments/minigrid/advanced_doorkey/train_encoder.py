from portable.option.memory import SetDataset 
from portable.option.ensemble.custom_attention import *
import matplotlib.pyplot as plt 
from tqdm import tqdm
import argparse

from torch.utils.tensorboard import SummaryWriter
import os 

files = [
    'resources/minigrid_images/adv_doorkey_random_states.npy',
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--feature_size", type=int)
    args = parser.parse_args()
    
    log_dir_base = "runs/advanced_doorkey/encoder"
    log_dir = os.path.join(log_dir_base, "0")
    x = 0
    while os.path.exists(log_dir):
        x += 1
        log_dir = os.path.join(log_dir_base, str(x))
    
    writer = SummaryWriter(log_dir=log_dir)
    dataset = SetDataset(batchsize=256, max_size=500000)
    dataset.add_true_files(files)
    
    feature_size = args.feature_size
    lr = 1e-4
    
    model = AutoEncoder(3, feature_size, image_height=128, image_width=128)
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

