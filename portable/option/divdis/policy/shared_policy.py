import torch
import torch.nn as nn

class SharedPolicy(nn.Module):
    def __init__(self, num_options, input_shape, num_actions, embed_dim=32):
        super().__init__()
        c, h, w = input_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.option_embed = nn.Embedding(num_options, embed_dim)
        cnn_output_dim = 16 * h * w
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, state, option_idx):
        x = self.cnn(state)
        o = self.option_embed(option_idx)
        x = torch.cat([x, o], dim=-1)
        return self.fc(x)

    def act(self, state, option_idx):
        with torch.no_grad():
            logits = self.forward(state.unsqueeze(0), option_idx.unsqueeze(0))
            return torch.argmax(logits, dim=-1).item()
