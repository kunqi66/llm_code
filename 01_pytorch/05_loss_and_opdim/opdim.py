import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ---- 常用优化器对比 ----
def run_optimizer(opt_name, lr=0.1, steps=100):
    x = torch.tensor([5.0], requires_grad=True)
    if opt_name == 'SGD':
        opt = torch.optim.SGD([x], lr=lr)
    elif opt_name == 'SGD+Momentum':
        opt = torch.optim.SGD([x], lr=lr, momentum=0.9)
    elif opt_name == 'Adam':
        opt = torch.optim.Adam([x], lr=lr)
    elif opt_name == 'AdamW':
        opt = torch.optim.AdamW([x], lr=lr, weight_decay=0.01)

    history = [x.item()]
    for _ in range(steps):
        loss = x ** 2        # f(x) = x^2，最小值在 x=0
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(x.item())
    return history

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
configs = [
    ('SGD',          0.1,  'tomato'),
    ('SGD+Momentum', 0.1,  'orange'),
    ('Adam',         0.5,  'steelblue'),
    ('AdamW',        0.5,  'seagreen'),
]
for name, lr, color in configs:
    hist = run_optimizer(name, lr)
    axes[0].plot(hist, color=color, lw=1.8, label=name)
    axes[1].plot([h**2 for h in hist], color=color, lw=1.8, label=name)

for ax, title in zip(axes, ['x 值变化（收敛到0）', '损失值变化']):
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlabel('步数')

plt.suptitle('不同优化器收敛速度对比', fontsize=13)
plt.tight_layout(); plt.show()

print('Adam/AdamW 自适应学习率，通常收敛更快')
print('SGD+Momentum 有时泛化更好，大模型训练常用')