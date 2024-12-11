import torch 
import torch.nn as nn 
import os 
import pickle 
from collections import deque 
import numpy as np
import logging

logger = logging.getLogger(__name__)

class NNClassifier():
    def __init__(self,
                 model_type,
                 image_height,
                 image_width,
                 num_channels,
                 use_gpu,
                 lr) -> None:
        assert model_type in ["cnn", "mlp"]
        if model_type == "cnn":
            self.model = CNNClassifier(image_height,
                                       image_width,
                                       num_channels)
        else:
            self.model = MLPClassifier(image_height,
                                       image_width,
                                       num_channels)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.use_gpu = use_gpu
        self.data = deque(maxlen=600)
        self.labels = deque(maxlen=600)
    
    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, "classifier.ckpt"))
        with open(os.path.join(path, 'data.pkl'), "wb") as f:
            pickle.dump(self.data, f)
        with open(os.path.join(path, 'labels.pkl'), "wb") as f:
            pickle.dump(self.labels, f)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, "classifier.ckpt")))
        with open(os.path.join(path, 'data.pkl'), "rb") as f:
            self.data = pickle.load(f)
        with open(os.path.join(path, 'labels.pkl'), "rb") as f:
            self.labels = pickle.dump(f)
    
    def move_to_gpu(self):
        if self.use_gpu:
            self.model.to("cuda")
    
    def move_to_cpu(self):
        self.model.to("cpu")
    
    def add_data(self, data, labels):
        assert len(data) == len(labels)
        
        self.data = self.data + deque(data)
        self.labels = self.labels + deque(labels)
    
    def train(self, epochs):
        self.move_to_gpu()
        for _ in range(epochs):
            for x, y in zip(self.data, self.labels):
                if type(x) is np.ndarray:
                    x = torch.from_numpy(x).float()
                if self.use_gpu:
                    x = x.to("cuda")
                x = x.unsqueeze(0)
                y = torch.tensor(y)
                y = y.unsqueeze(0)
                if self.use_gpu:
                    y = y.to("cuda")
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        accuracies = []
        for x, y in zip(self.data, self.labels):
            with torch.no_grad():
                if type(x) is np.ndarray:
                    x = torch.from_numpy(x).float()
                if self.use_gpu:
                    x = x.to("cuda")
                x = x.unsqueeze(0)
                y = torch.tensor(y)
                y = y.unsqueeze(0)
                if self.use_gpu:
                    y = y.to("cuda")
                pred = self.model(x)
                accuracies.append(int(torch.argmax(pred) == y))
        
        final_accuracy = np.mean(accuracies)
        
        logger.info("[nn classifier] trained model for {} epochs. Final Accuracy: {}".format(epochs, final_accuracy))
        self.move_to_cpu()
    
    def predict(self, x):
        if self.use_gpu:
            x.to("cuda")
        return self.model(x)

class PrintLayer(torch.nn.Module):
    # print input. For debugging
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        print(x.shape)
        
        return x

class MLPClassifier(nn.Module):
    def __init__(self,
                 image_height,
                 image_width,
                 num_channels):
        super().__init__()
        
        input_dim = image_height*image_width*num_channels
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.ReLU(),
            nn.Linear(input_dim//4, 2),
            nn.Softmax(-1),
        )
    
    def forward(self, x):
        
        return self.network(x)

class CNNClassifier(nn.Module):
    def __init__(self,
                 image_height,
                 image_width,
                 num_input_channels):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(num_input_channels, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.LazyLinear(1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Softmax(-1)
        )
    
    
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        return self.network(x)


