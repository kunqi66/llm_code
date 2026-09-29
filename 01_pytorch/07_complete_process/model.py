from torch import nn
import torch


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.laryer1 = nn.Linear(8, 32)
        self.laryer2 = nn.Linear(32, 16)
        self.laryer3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        
        
    def forward(self, X):
        X = self.laryer1(X)
        X = self.relu(X)
        X = self.laryer2(X)
        X = self.relu(X)
        X = self.laryer3(X)
        return X
    
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = MyModel().to(device=device)


