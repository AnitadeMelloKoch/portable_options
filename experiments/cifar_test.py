import torchvision
from portable.option.ensemble.multiheaded_attention import ViT
import torch

def get_dataset():
    dataset = torchvision.datasets.CIFAR10(
        "resources/cifar",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            
        ])
    )
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=128,
                                             shuffle=True)
    
    return dataloader

global device 
device = torch.device("cuda")

vit = ViT(
    in_channel=3,
    patch_size=4,
    feature_dim=768,
    img_size=32,
    depth=12,
    n_classes=10,
    **{
        "attention_num": 12,
        "dropout_prob": 0.,
        "forward_expansion": 4,
        "forward_dropout": 0.
    }
).to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vit.parameters(),lr=0.001)

def train(model, dataset, criterion, optim, epochs):
    model.train()
    for epoch in range(epochs):
        loss = 0
        acc = 0
        counter = 0
        for x, y in dataset:
            x = x.to(device)
            y = y.to(device)
            pred_y = model(x)
            b_loss = criterion(pred_y, y)
            pred_class = torch.argmax(pred_y, dim=1).detach()
            acc += (torch.sum(pred_class==y).item())/len(y)
            counter += 1
            b_loss.backward()
            optim.step()
            optim.zero_grad()
            loss += b_loss.item()
        
        print("epoch {} = loss: {:.2f} acc: {:.2f}".format(epoch, loss/counter, acc/counter))

train(
    vit,
    get_dataset(),
    loss,
    optimizer,
    2000
)
