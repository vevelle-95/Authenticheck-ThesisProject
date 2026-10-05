import torch.nn as nn

class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)