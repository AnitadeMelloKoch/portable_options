from portable.option.memory import SetDataset
from portable.option.ensemble.custom_attention import *
import matplotlib.pyplot as plt 

encoder_dir = "runs/custom_attention_test/encoder/2/encoder.ckpt"

initiation_positive_files = [
    'resources/minigrid_images/doorkey_getkey_0_initiation_image_positive.npy',
    'resources/minigrid_images/doorkey_getkey_1_initiation_image_positive.npy',
    'resources/minigrid_images/doorkey_getkey_2_initiation_image_positive.npy',
]

initiation_negative_files = [
    'resources/minigrid_images/doorkey_getkey_0_initiation_image_negative.npy',
    'resources/minigrid_images/doorkey_getkey_1_initiation_image_negative.npy',
    'resources/minigrid_images/doorkey_getkey_2_initiation_image_negative.npy',
]

num_channels = 6

dataset = SetDataset(batchsize=16)
dataset.add_true_files(initiation_positive_files)

model = AutoEncoder(6)
model.load_state_dict(torch.load(encoder_dir))
device = torch.device("cuda")
model.to(device)

with torch.no_grad():

    x, _ = dataset.get_batch()
    x = x.to(device)

    embedding = model.encoder(x)
    embedding = embedding.view(-1, num_channels, 128)

    mask = torch.from_numpy(np.array(
        [1, 1, 1, 0, 0, 0]
    )).to(device)
    mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(-1)

    embedding = embedding*mask
    
    embedding = embedding.view(-1, num_channels*128)


    embedding = model.decoder_linear(embedding)
    embedding = embedding.view(-1, 4*16, 21, 21)
    output = model.decoder(embedding)
    
    # output2 = model(x)
    
    fig, axes = plt.subplots(nrows=3, ncols=6)
    for i in range(10):
        sample = x[i]
        with torch.no_grad():
            output2 = model(sample.unsqueeze(0))
            for idx in range(6):
                axes[0,idx].set_axis_off()
                axes[1,idx].set_axis_off()
                axes[2,idx].set_axis_off()
                axes[0,idx].imshow(sample[idx].cpu().numpy(), cmap='gray')
                axes[1,idx].imshow(output2[0,idx,...].cpu().numpy(), cmap='gray')
                axes[2,idx].imshow(output[i,idx,...].cpu().numpy(), cmap='gray')
        
            plt.show(block=False)
            input("continue")

    plt.close(fig)













