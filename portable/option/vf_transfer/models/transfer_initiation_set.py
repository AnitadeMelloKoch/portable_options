import torch.nn as nn
import torch 
import gin 
import torch.optim as optim
import torch.nn.functional as F
from portable.option.memory.set_dataset import SetDataset
import logging 
logger = logging.getLogger(__name__)

class TransferInitiationModel(nn.Module):
    def __init__(self,
                 max_skill_num=50,
                 embedding_dim=256):
        super().__init__()
        
        self.skill_embedding = nn.Embedding(max_skill_num, embedding_dim)
        
        self.image_body = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.head = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        skill_embed = self.skill_embedding(x)
        image_embed = self.image_body(x)
        embed = torch.concat((skill_embed, image_embed), dim=1)
        
        out = self.head(embed)
        
        return F.sigmoid(out)
        
class TransferInitiationSet():
    def __init__(self,
                 batchsize=16,
                 max_size=1e6,
                 max_skill_num=50,
                 embedding_dim=256):
        self.device = None
        self.loss = None 
        self.optim = None
        
        self.data = SetDataset(batchsize=batchsize,
                               max_size=max_size,
                               unlabelled_batchsize=None)
        
        self.model = TransferInitiationModel(max_skill_num=max_skill_num,
                                             embedding_dim=embedding_dim)
        
    def store_data(self, positive_states, negative_states):
        self.data.add_true_data(positive_states)
        self.data.add_false_data(negative_states)
        
    def train(self, epochs):
        
        logging.info("Training initiation set...")
        
        for epoch in range(epochs):
            self.data.shuffle()
            counter = 0.0
            init_loss = 0.0
            init_acc = 0.0
            
            for _ in range(self.data.num_batches):
                counter += 1
                x, y = self.data.get_batch()
                x = x.to(self.device)
                pred_y = self.model(x)
                
                loss = self.loss(pred_y, y)
                
                loss.backward()
                init_loss += loss.item()
                self.optim.step()
                self.optim.zero_grad()
                
                pred_class = torch.argmax(pred_y, dim=1).detach()
                init_acc += (torch.sum(pred_class==y).item())/len(y)
                
            
            logger.info(f"Epoch: {epoch} Initiation loss: {init_loss/counter} Initiation accuracy: {init_acc/counter}")
        
    def query(self, state):
        state = state.to()
        
        return self.model(state)
            
            
            
        
        
        