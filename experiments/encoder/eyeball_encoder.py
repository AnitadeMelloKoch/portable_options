import numpy as np 
from portable.option.ensemble.custom_attention import AutoEncoder
import torch
import matplotlib.pyplot as plt
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
from experiments.minigrid.utils import environment_builder

def concatenate(arr1, arr2):

        if len(arr1) == 0:
            return arr2
        else:
            return np.concatenate((arr1, arr2), axis=0)

def load_data(file_names):
    all_data = np.array([])
    for file in file_names:
        data = np.load(file)
        data = np.squeeze(data)
        all_data = concatenate(all_data, data)
    
    return all_data

autoencoder = AutoEncoder(num_input_channels=3,
                          feature_size=500,
                          image_height=64,
                          image_width=64).to('cuda')

autoencoder.load_state_dict(torch.load("resources/encoders/doorkey_8x8/encoder.ckpt"))


positive_files = [
    'resources/minigrid_images/adv_doorkey_8x8_getredkey_doorred_0_initiation_positive.npy',
    'resources/minigrid_images/adv_doorkey_8x8_random_states_red.npy'
]
negative_files = [
    'resources/minigrid_images/adv_doorkey_8x8_getredkey_doorred_0_initiation_negative.npy',
    'resources/minigrid_images/adv_doorkey_8x8_random_states_no_red.npy',
]

env1 = environment_builder('AdvancedDoorKey-8x8-v0', 
                          seed=3, grayscale=False)
env1 = AdvancedDoorKeyPolicyTrainWrapper(env1,
                                         door_colour="red")

env2 = environment_builder('AdvancedDoorKey-8x8-v0', 
                          seed=3, grayscale=False)
env2 = AdvancedDoorKeyPolicyTrainWrapper(env2,
                                         door_colour="red",
                                         key_collected=True)

state1,_ = env1.reset(3)
state2,_ = env2.reset(3)

plt.imshow(np.transpose(state1, (1,2,0)))
plt.savefig("runs/eyeball_embeddings/with_key.png")
plt.close()

plt.imshow(np.transpose(state2, (1,2,0)))
plt.savefig("runs/eyeball_embeddings/no_key.png")
plt.close()

state1 = state1.to("cuda").float().unsqueeze(0)
state2 = state2.to("cuda").float().unsqueeze(0)

embedding1 = autoencoder.feature_extractor(state1)
print(embedding1.shape)
embedding2 = autoencoder.feature_extractor(state2)
print(embedding2.shape)
diff = embedding1-embedding2
print(diff.shape)

plt.bar(range(0,500), embedding1.squeeze().cpu().numpy())
plt.savefig("runs/eyeball_embeddings/emb1.png")
plt.close()

plt.bar(range(0,500), embedding2.squeeze().cpu().numpy())
plt.savefig("runs/eyeball_embeddings/emb2.png")
plt.close()

plt.bar(range(0,500), diff.squeeze().cpu().numpy())
plt.savefig("runs/eyeball_embeddings/diff.png")
plt.close()

# positive_samples = load_data(positive_files)
# negative_samples = load_data(negative_files)

# print(positive_samples.shape)
# print(negative_samples.shape)


# positive_samples = torch.from_numpy(positive_samples)
# positive_samples = positive_samples.to("cuda").float()

# negative_samples = torch.from_numpy(negative_samples)
# negative_samples = negative_samples.to("cuda").float()

# positive_embedding = autoencoder.feature_extractor(positive_samples)
# negative_embedding = autoencoder.feature_extractor(negative_samples)
# print(positive_embedding.shape)
# print(negative_embedding.shape)

# positive_std, positive_mean = torch.std_mean(positive_embedding, dim=0)
# negative_std, negative_mean = torch.std_mean(negative_embedding, dim=0)
# print(positive_mean.shape)
# print(positive_std.shape)
# print(negative_mean.shape)

# diff = positive_mean-negative_mean

# plt.bar(range(0,500), diff.cpu().numpy())
# plt.savefig("runs/eyeball_embeddings/diff.png")
# plt.close()


# plt.bar(range(0,500), positive_mean.cpu().numpy())
# # plt.errorbar(range(0,500), 
# #              positive_mean.cpu().numpy(), 
# #              positive_std.cpu().numpy(),
# #              fmt="o",
# #              color="r")
# plt.savefig("runs/eyeball_embeddings/pos_mean.png")
# plt.close()

# plt.bar(range(0,500), negative_mean.cpu().numpy())
# # plt.errorbar(range(0,500), negative_mean.cpu().numpy(), negative_std.cpu().numpy())
# plt.savefig("runs/eyeball_embeddings/neg_mean.png")
# plt.close()

# plt.bar(range(0,500), positive_mean.cpu().numpy())
# plt.bar(range(0,500), negative_mean.cpu().numpy())
# plt.savefig("runs/eyeball_embeddings/combined_mean.png")
# plt.close()