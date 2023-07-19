from portable.option.memory import SetDataset
from portable.option.ensemble.custom_attention import *

from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

initiation_positive_files = [
    'resources/minigrid_images/doorkey_getkey_0_initiation_image_positive.npy',
]

initiation_negative_files = [
    'resources/minigrid_images/doorkey_getkey_0_initiation_image_negative.npy',
]

test_positive_files = [
    'resources/minigrid_images/doorkey_getkey_1_initiation_image_positive.npy',
    'resources/minigrid_images/doorkey_getkey_2_initiation_image_positive.npy',
]

test_negative_files = [
    'resources/minigrid_images/doorkey_getkey_1_initiation_image_negative.npy',
    'resources/minigrid_images/doorkey_getkey_2_initiation_image_negative.npy',
]

dataset = SetDataset(batchsize=16)

dataset.add_true_files(initiation_positive_files)
dataset.add_false_files(initiation_negative_files)

test_dataset = SetDataset(batchsize=16)

test_dataset.add_true_files(test_positive_files)
test_dataset.add_false_files(test_negative_files)


model = AttentionEnsembleII(num_attention_heads=8,
                            num_features=6,
                            num_classes=2)

criterion = torch.nn.CrossEntropyLoss()
optimizers = [
    torch.optim.Adam(model.attentions[idx].parameters(), lr=1e-4) for idx in range(8)
]

device = torch.device("cuda")
model.to(device)

for epoch in range(5000):
    loss = np.zeros(8)
    classifier_losses = np.zeros(8)
    classifier_acc = np.zeros(8)
    div_losses = np.zeros(8)
    acc = 0
    counter = 0
    for b_idx in range(dataset.num_batches):
        counter += 1
        x, y = dataset.get_batch()
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x)
        for att_idx in range(8):
            b_loss = criterion(pred_y[att_idx], y)
            pred_class = torch.argmax(pred_y[att_idx], dim=1).detach()
            classifier_losses[att_idx] += b_loss.item()
            div_loss = divergence_loss(model.get_attention_masks(), att_idx)
            div_losses[att_idx] += div_loss.item()
            regulariser_loss = l1_loss(model.get_attention_masks(), att_idx)
            b_loss += div_loss
            b_loss += regulariser_loss
            classifier_acc[att_idx] += (torch.sum(pred_class==y).item())/len(y)
            b_loss.backward()
            optimizers[att_idx].step()
            optimizers[att_idx].zero_grad()
            loss[att_idx] += b_loss.item()
    
    for idx in range(8):
        print("att {} epoch {} = class loss: {:.2f} div loss: {:.2f} total loss: {:.2f} acc: {:.2f}".format(idx, 
                                                                  epoch, 
                                                                  classifier_losses[idx]/counter,
                                                                  div_losses[idx]/counter,
                                                                  loss[idx]/counter,
                                                                  classifier_acc[idx]/counter))
    
    if epoch%100 == 0:
        accuracies = np.zeros(8)
        counter = 0
        for b_idx in range(test_dataset.num_batches):
            # print(b_idx)
            with torch.no_grad():
                x, y = test_dataset.get_batch()
                x = x.to(device)
                y = y.to(device)
                pred_y = model(x)
                counter += 1
                for att_idx in range(8):
                    pred_class = torch.argmax(pred_y[att_idx], dim=1).detach()
                    accuracies[att_idx] += torch.sum(pred_class==y).item()/len(y)
                    # print("prediction",pred_class)
                    # print("true",y)

        print("****************************")
        for idx in range(8):
            print("attention {} test accuracy {}".format(idx,
                                                        accuracies[idx]/counter))
        print("****************************")
    
    
    
    print("========================================")

for idx in range(8):
    print("attention {}".format(idx))
    print(model.attentions[idx].attention.mask())

accuracies = np.zeros(8)
counter = 0
for b_idx in range(test_dataset.num_batches):
    # print(b_idx)
    with torch.no_grad():
        x, y = test_dataset.get_batch()
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x)
        counter += 1
        for att_idx in range(8):
            pred_class = torch.argmax(pred_y[att_idx], dim=1).detach()
            accuracies[att_idx] += torch.sum(pred_class==y).item()/len(y)
            # print("prediction",pred_class)
            # print("true",y)

for idx in range(8):
    print("attention {} test accuracy {}".format(idx,
                                                 accuracies[idx]/counter))
            


