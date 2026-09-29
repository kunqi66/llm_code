import torch
from torch.utils.data import Dataset,DataLoader,random_split
from torch import nn
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(42)

X = torch.randn(500, 8)
y = (X[:,0:4].sum(dim=1) > 0).to(dtype=torch.long)

class MyData(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return X[index] , y[index]
    

data = MyData(X, y)

train_data, val_data = random_split(data, [0.7,0.3])
train_data_loader = DataLoader(train_data,shuffle=True,batch_size=32)
val_data_loader = DataLoader(val_data, shuffle=True, batch_size=32)



        


