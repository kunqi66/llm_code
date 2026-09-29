import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 构建数据集的关键是自定义  __len__ 和 __getitem__
class CustomDataset(Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,inx):
        return self.x[inx], self.y[inx]
    
    
np.random.seed(42)
X = np.random.randn(100, 4).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

dataset = CustomDataset(X, y)
print('数据集大小:', len(dataset))
print('第0条样本:')
x0, y0 = dataset[0]

print('  x:', x0.tolist())
print('  y:', y0.item())



# 使用DataLoader批量加载


from torch.utils.data import random_split

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)


print('训练集批次数:', len(train_loader))
print('验证集批次数:', len(val_loader))

for X_batch, y_batch in train_loader:
    print('\n一个 batch:')
    print('X_batch shape:', X_batch.shape)   # (16, 4)
    print('y_batch shape:', y_batch.shape)   # (16,)
    break



print('\nDataLoader 常用参数：')
print('  batch_size: 每批样本数')
print('  shuffle:    每个 epoch 是否打乱顺序（训练集开，验证集关）')
print('  num_workers: 并行加载的子进程数（Windows 建议0，Linux 可设4）')
print('  drop_last:  最后一批样本数不足时是否丢弃')