from portable.option.memory import SetDataset
from portable.option.ensemble.custom_attention import *
import matplotlib.pyplot as plt 

import copy

from torch.utils.tensorboard import SummaryWriter
import os 

initiation_positive_files = [
    'resources/factored_minigrid_images/doorkey_getkey_0_initiation_image_positive.npy',
    'resources/factored_minigrid_images/doorkey_getkey_1_initiation_image_positive.npy',
    'resources/factored_minigrid_images/doorkey_getkey_2_initiation_image_positive.npy',
]

initiation_negative_files = [
    'resources/factored_minigrid_images/doorkey_getkey_0_initiation_image_negative.npy',
    'resources/factored_minigrid_images/doorkey_getkey_1_initiation_image_negative.npy',
    'resources/factored_minigrid_images/doorkey_getkey_2_initiation_image_negative.npy',
]

other = [
    'resources/factored_minigrid_images/doorkey_start.npy',
    'resources/factored_minigrid_images/doorkey_start_2.npy',
    'resources/factored_minigrid_images/doorkey_no_key.npy',
    'resources/factored_minigrid_images/doorkey_open_door.npy',
    
]

log_dir_base = "runs/custom_attention_test/encoder"
log_dir = os.path.join(log_dir_base, "0")
x = 0
while os.path.exists(log_dir):
    x += 1
    log_dir = os.path.join(log_dir_base, str(x))

writer = SummaryWriter(log_dir=log_dir)

dataset = SetDataset(batchsize=64)
dataset.add_true_files(initiation_positive_files)
dataset.add_true_files(initiation_negative_files)
dataset.add_true_files(other)

model = AutoEncoder(6, 8, 1000)
mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda")
model.to(device)

test_x = None
train_x = None

for epoch in range(5000):
    dataset.shuffle()
    loss = 0
    kl_losses = 0
    mse_losses = 0
    counter_train = 0
    for b_idx in range(dataset.num_batches):
        counter_train += 1
        x, _ = dataset.get_batch()
        print(x.shape)
        x = x.to(device)
        train_x = x
        pred = model(x)
        mse_loss = mse(pred, x)
        mse_losses += mse_loss.item()
        b_loss = mse_loss
        # b_loss += encoder_loss(x, pred)
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
        fig, axes = plt.subplots(nrows=2, ncols=6)
        sample = train_x[i]
        with torch.no_grad():
            output = model(sample.unsqueeze(0))
            for idx in range(6):
                axes[0,idx].set_axis_off()
                axes[1,idx].set_axis_off()
                axes[0,idx].imshow(sample[idx].cpu().numpy(), cmap='gray')
                axes[1,idx].imshow(output[0,idx,...].cpu().numpy(), cmap='gray')
        
            fig.savefig(os.path.join(log_dir, "{}.png".format(i)))

        plt.close(fig)
