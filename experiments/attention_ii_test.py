from portable.option.memory import PositionSet, factored_minigrid_formatter
from portable.option.ensemble.custom_attention import *

initiation_positive_files = [
    'resources/minigrid_images/doorkey_getkey_0_initiation_loc_positive.npy',
    'resources/minigrid_images/doorkey_getkey_1_initiation_loc_positive.npy',
    'resources/minigrid_images/doorkey_getkey_2_initiation_loc_positive.npy',
]

initiation_negative_files = [
    'resources/minigrid_images/doorkey_getkey_0_initiation_loc_negative.npy',
    'resources/minigrid_images/doorkey_getkey_1_initiation_loc_negative.npy',
    'resources/minigrid_images/doorkey_getkey_2_initiation_loc_negative.npy',
]

dataset = PositionSet(batchsize=16,
                      data_formatter=factored_minigrid_formatter)

dataset.add_true_files(initiation_positive_files)
dataset.add_false_files(initiation_negative_files)

model = AttentionSetII(num_features=6,
                       num_classes=2)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device("cuda")
model.to(device)


for epoch in range(100):
    loss = 0
    acc = 0
    counter = 0
    for b_idx in range(dataset.num_batches):
        counter += 1
        x, y = dataset.get_batch()
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x)
        b_loss = criterion(pred_y, y)
        pred_class = torch.argmax(pred_y, dim=1).detach()
        # print("input", x)
        # print("pred",pred_class)
        # print("target",y)
        acc += (torch.sum(pred_class==y).item())/len(y)
        b_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss += b_loss.item()
    print("epoch {} = loss: {:.6f} acc: {:.2f}".format(epoch, loss/counter, acc/counter))
    print(model.attention.mask())
