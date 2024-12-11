import torch 
import torch.nn as nn
import torch.nn.functional as F

class OptionDQN(nn.Module):
    def __init__(self,
                 height,
                 width,
                 channel_num,
                 action_vector_size,
                 num_options,
                 hidden_size=128):
        super().__init__()
        
        self.conv_network = nn.Sequential(
            nn.Conv2d(channel_num, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        def conv2d_size_out(size, kernel_size=5,stride=2):
            return (size - (kernel_size-1)-1) // stride + 1
        
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        
        linear_input_size = conv_w*conv_h*32
        
        self.image_fc = nn.Linear(linear_input_size, hidden_size)
        self.action_fc = nn.Linear(action_vector_size, hidden_size)
        
        self.head = nn.Linear(2*hidden_size, num_options)
        
    def forward(self, action_output, x):
        
        x = self.conv_network(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.image_fc(x))
        
        action_output = torch.tensor(action_output).float().to("cuda")
        
        action_output = F.relu(self.action_fc(action_output))
        concat_input = torch.cat((x, action_output), dim=1)
        
        return F.relu(self.head(concat_input))
        
        
        
