import torch
from torch import nn
import config
from model import model
from data_load import train_data_loader,val_data_loader
import matplotlib.pyplot as plt


loss_fn = nn.CrossEntropyLoss()
optimzer = torch.optim.Adam(model.parameters(),lr=config.lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, train_dataloader, loss_fn, optimzer, device):
    # 忘记着重
    model.train()
    total_loss , correct = 0,0
    
    for X_batch , y_batch in train_dataloader:
        X_batch,y_batch = X_batch.to(device), y_batch.to(device)
        pre = model(X_batch)
        loss = loss_fn(pre, y_batch)
        
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        
        total_loss += loss.item()
        correct   +=  (pre.argmax(1) == y_batch).sum().item()

    return total_loss / len(train_dataloader.dataset), correct / len(train_dataloader.dataset)

def evaluate(model, val_dataloader, loss_fn, device):
    model.eval()
    total_loss,correct = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pre = model(X_batch)
            loss = loss_fn(pre, y_batch)
            total_loss += loss.item()
            correct += (pre.argmax(1) == y_batch).sum().item()
    
    return total_loss / len(val_dataloader.dataset), correct / len(val_dataloader.dataset)
    

def train(loss_fn, optimzer, model, train_dataloader, val_dataloader, epoch, device):
    # 用于可视化绘图
    history = {"train_loss" : [], "val_loss" : [], "train_acc" : [], "val_acc" : []}
    
    for i in range(epoch):
        tr_loss , tr_acc = train_one_epoch(model, train_dataloader, loss_fn, optimzer, device)
        va_loss , va_acc = evaluate(model, val_dataloader, loss_fn, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_loss"].append(va_loss)
        
        if (i + 1) % 5 == 0:
            print(f"epoch : {i+1} | ",
                  f"\ntrain : train_loss : {tr_loss} , train_acc : {tr_acc} | ",
                  f"\nevaluare : val_loss : {va_loss}, val_acc  : {va_acc}")
    return history
            
if __name__ == "__main__":
    history = train(loss_fn,optimzer,model,train_data_loader,val_data_loader,config.epoch,device)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    ax1.plot(history['train_loss'], label = '训练')
    ax1.plot(history['val_loss'], label='验证')
    ax1.set_title('损失')
    ax1.legend()
    ax1.grid(True, alpha =  0.3)
    ax2.plot(history['train_acc'], label='训练')
    ax2.plot(history['val_acc'], label='验证')
    ax2.set_title('准确率'); ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
