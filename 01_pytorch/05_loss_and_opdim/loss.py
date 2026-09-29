import torch
import torch.nn as nn

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- 分类任务 ----

# CrossEntropyLoss（最常用：包含 Softmax + NLLLoss）
ce_loss = nn.CrossEntropyLoss()
logits  = torch.tensor([[2.0, 1.0, 0.5],   # 样本1的 logit
                         [0.5, 2.0, 1.0]])  # 样本2的 logit
targets = torch.tensor([0, 1])              # 真实类别
print('CrossEntropyLoss:', ce_loss(logits, targets).item())

# BCEWithLogitsLoss（二分类，输入不需要 sigmoid）
bce = nn.BCEWithLogitsLoss()
logits_bin = torch.tensor([2.0, -1.0, 0.5])
labels_bin = torch.tensor([1.0,  0.0, 1.0])
print('BCEWithLogitsLoss:', bce(logits_bin, labels_bin).item())

# ---- 回归任务 ----
y_pred = torch.tensor([1.5, 2.0, 3.5])
y_true = torch.tensor([1.0, 2.5, 3.0])

print('MSELoss:    ', nn.MSELoss()(y_pred, y_true).item())
print('L1Loss:     ', nn.L1Loss()(y_pred, y_true).item())
print('HuberLoss:  ', nn.HuberLoss()(y_pred, y_true).item())  # L1+L2 混合