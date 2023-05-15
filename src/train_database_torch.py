import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from early_stop import early_stop
from tqdm import tqdm

s10 = './train_data_0130/train_data_0130_s10_valid_100000.txt'

### Config
BOARD_SIZE = 8
MAXLEN = 128
BATCH_SIZE = 32
EPOCHS = 10000
LR = 1e-3
EARLY_STOP = early_stop(save_path='E:/hsuanchia_e/Othello_2022/s10_allvalid_multilabel_weights/torch/', mode='min', monitor='val_loss', patience=5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

### Preprocess data --> for cnn2d
def build_data_2d(path, aug=False):
    board, labels = [], []
    f = open(path)
    for l in f.readlines():
        cur_board = [[],[],[],[],[],[],[],[]]
        label = []
        label.extend([0] * (BOARD_SIZE * BOARD_SIZE))
        init_label = label.copy()
        tmp = l.split(" [")
        b_tmp = tmp[0].split(",")[:-1]
        l_tmp = tmp[1].split(",")
        # print(tmp)
        for i in range(0,len(b_tmp)):
            row = b_tmp[i].split(" ")
            for j in row[1:]:
                if(j[0] == '1'):
                    cur_board[i].append(1)
                if(j[0] == '0'):
                    cur_board[i].append(0)
                if(j[0] == '-'):
                    cur_board[i].append(-1)
        l_tmp = l_tmp[:-1]
        # print(l_tmp)
        for y in l_tmp:
            label[int(y[1]) * BOARD_SIZE + int(y[3])] = 1
        board.append(cur_board)
        labels.append(label)
        label = init_label.copy()
        if aug:
            ### Board rotate 90, 180, 270 degree
            for rotate in range(3):
                new_board = [[],[],[],[],[],[],[],[]]
                for i in range(len(cur_board)):
                    for j in range(len(cur_board[i])):
                        new_board[j].append(cur_board[i][j])
                for i in range(len(new_board)):
                    new_board[i].reverse()
                ### Position rotate -> y pop to x, y = BOARD_SIZE-1-x
                for y in l_tmp:
                    new_x = int(y[3])
                    new_y = int(BOARD_SIZE - 1 - int(y[1]))
                    label[new_x * BOARD_SIZE + new_y] = 1
                    board.append(new_board)
                    labels.append(label)
    
    return board, labels

### Preprocess data --> for cnn1d
def build_data_1d(path):
    board, labels = [], []
    f = open(path)
    for l in f.readlines():
        cur_board = [[],[],[],[],[],[],[],[]]
        label = []
        label.extend([0] * (BOARD_SIZE * BOARD_SIZE))
        tmp = l.split(" [")
        b_tmp = tmp[0].split(",")[:-1]
        l_tmp = tmp[1].split(",")
        # print(tmp)
        for i in range(0,len(b_tmp)):
            row = b_tmp[i].split(" ")
            for j in row[1:]:
                if(j[0] == '1'):
                    cur_board[i].append(1)
                if(j[0] == '0'):
                    cur_board[i].append(0)
                if(j[0] == '-'):
                    cur_board[i].append(-1)
        tmp_board = []
        for i in range(len(cur_board)):
            tmp_board.extend(cur_board[i])
        l_tmp = l_tmp[:-1]
        # print(l_tmp)
        for y in l_tmp:
            label[int(y[1]) * BOARD_SIZE + int(y[3])] = 1
        board.append(tmp_board)
        labels.append(label)

    return board, labels

### Othello Board dataset
class OthelloBoard(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, ind):
        train_x = np.expand_dims(self.x[ind], axis=1)
        train_y = self.y[ind]
        return torch.tensor(train_x), torch.tensor(train_y)

### Model --> CNN1D
class CNN1D(nn.Module):
    def __init__(self) -> None:
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.flat  = nn.Flatten()
        self.drop = nn.Dropout(0.2)
        
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.linear1 = nn.Linear(1008, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 64)

    def forward(self, x):
        x = x.transpose(1, 2) ### Difference than keras
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = self.linear3(x)
        x = nn.functional.sigmoid(x)

        return x

### Customize loss function
class OthelloLoss(nn.Module):
    def __init__(self):
        super(OthelloLoss, self).__init__()

    def forward(self, pred, label, th):
        BCE_loss = nn.BCELoss()(pred, label)
        pred = torch.where(pred > th, True, False)
        label = torch.where(label == 1.0, True, False)
        # for i in range(len(pred)):
        xor = torch.bitwise_xor(pred, label)
        wrong_pred = torch.bitwise_and(pred, xor)
        miss_pred = torch.bitwise_xor(xor, wrong_pred)
        tmp1 = torch.where(miss_pred == True, 1, 0)
        miss_sum = torch.sum(tmp1, 1)
        tmp2 = torch.where(wrong_pred == True, 1, 0)
        wrong_sum = torch.sum(tmp2, 1)
        win_pos = torch.sum(torch.where(label == True, 1, 0), 1)

        wrong_loss = torch.div(torch.div(wrong_sum, len(wrong_sum)), (BOARD_SIZE)) 
        wrong_loss = torch.nan_to_num(wrong_loss, nan=0.0)
        wrong_batch_loss = torch.div(torch.sum(wrong_loss), len(wrong_loss))
        miss_loss = torch.div(torch.div(miss_sum, len(miss_sum)), win_pos) 
        miss_loss = torch.nan_to_num(miss_loss, nan=0.0)
        miss_batch_loss = torch.div(torch.sum(miss_loss), len(miss_loss))

        # print(f"BCE_Loss: {BCE_loss}, Wrong_Loss: {wrong_batch_loss}, Miss_Loss: {miss_batch_loss}")

        total_loss = BCE_loss + wrong_batch_loss + miss_batch_loss
        # print(total_loss)

        return total_loss

### Customize accuracy function
def evaluate_accuracy(pred, label):
    all_zero = [0.0] * 64
    # print(pred)
    # print(label)
    if pred == all_zero and label != all_zero:
        return 0
    
    for i in range(len(pred)):
        if pred[i] == 1.0 and label[i] == 0.0: ### Always here!
            return 0
    return 1

### Training stage
def train(train_loader, val_loader, model, batch_size, optimizer, loss_func, acc_func, epoch):
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    best_acc, best_loss = 0, 0
    best_val_acc, best_val_loss = 0, 0
    for num_epoch in range(epoch):
        train_avg_loss = 0                #每個epoch的loss
        train_avg_acc = 0                 #每個epoch的acc
        total_acc = 0
        for step, (train_x, train_y) in tqdm(enumerate(train_loader)):
            #確保每一batch都能進入model.train模式
            model.train()
            #放置gpu訓練
            train_x = train_x.to(device, dtype=torch.float)
            train_y = train_y.to(device, dtype=torch.float)
            #img經過nural network卷積後的預測(前向傳播),跟答案計算loss 
            pred = model(train_x)
            criterion = loss_func()
            loss = criterion(pred, train_y, th=0.5)
            #優化器的gradient每次更新要記得初始化,否則會一直累積
            optimizer.zero_grad()
            #反向傳播偏微分,更新參數值
            loss.backward()
            #更新優化器
            optimizer.step()

            #累加每個batch的loss後續再除step數量
            train_avg_loss += loss.item()
            
            ### Evaluate accuracy
            # pred = np.where(pred.detach().cpu().numpy() > 0.5, 1.0, 0.0).tolist()
            ans = pred.argmax(dim=1)
            train_y = train_y.detach().cpu().numpy().tolist()
            hit = 0
            # print(ans)
            # print(train_y)
            for i in range(len(ans)):
                if train_y[i][int(ans[i])] == 1.0:
                        hit += 1
                # hit += acc_func(pred[i], train_y[i])
            batch_acc = hit / len(pred)
            total_acc += batch_acc

        val_avg_loss,val_avg_acc = validation(val_loader=val_loader, model=model, loss_func=loss_func, acc_func=acc_func)
        
        train_avg_loss = round(train_avg_loss/len(train_loader),4)   #該epoch每個batch累加的loss平均
        train_avg_acc = round(total_acc/len(train_loader),4)         #該epoch的acc平均

        train_loss.append(train_avg_loss)
        train_acc.append(train_avg_acc)
        val_loss.append(val_avg_loss)
        val_acc.append(val_avg_acc)

        print('Epoch: {} | train_loss: {} | train_acc: {}% | val_loss: {} | val_acc: {}%'\
              .format(num_epoch, train_avg_loss,round(train_avg_acc*100,4),val_avg_loss,round(val_avg_acc*100,4)))
        
        ### early stop
        performance_value = [num_epoch, train_avg_loss, round(train_avg_acc*100,4), val_avg_loss, round(val_avg_acc*100,4)]
        EARLY_STOP(val_avg_loss, model=model, performance_value=performance_value)
        
        if EARLY_STOP.early_stop:
            print('Early stopping')
            break    

    return train_loss, train_acc, val_loss, val_acc 

### Validation stage
def validation(val_loader, model, loss_func, acc_func):
    val_avg_loss, val_avg_acc = 0, 0
    total_acc = 0
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for val_x, val_y in tqdm(val_loader):
            val_x = val_x.to(device, dtype=torch.float)
            val_y = val_y.to(device, dtype=torch.float)

            pred = model(val_x)

            criterion = loss_func()
            loss = criterion(pred, val_y, th=0.5)
            #累加每個batch的loss後續再除step數量
            val_avg_loss += loss.item()

            ### Evaluate accuracy
            # pred = pred.detach().cpu().numpy().tolist()
            val_y = val_y.detach().cpu().numpy().tolist()
            hit = 0
            ans = pred.argmax(dim=1)
            for i in range(len(ans)):
                if val_y[i][int(ans[i])] == 1.0:
                        hit += 1
                # hit += acc_func(pred[i], val_y[i])
            batch_acc = hit / len(pred)
            total_acc += batch_acc

        val_avg_loss = round(val_avg_loss/len(val_loader),4)
        val_avg_acc = round(total_acc/len(val_loader),4)        

    return val_avg_loss, val_avg_acc

if __name__ == '__main__':
    x, y = build_data_1d(s10)
    total_len = len(x)
    print(f'Total data length: {total_len}')
    data_len = int(total_len * 0.8)
    train_x, train_y = x[:data_len], y[:data_len]
    val_x, val_y = x[data_len:], y[data_len:]
    print(f'Train length: {len(train_x)}')
    print(f'Val length: {len(val_x)}')

    train_ds = OthelloBoard(train_x, train_y)
    val_ds = OthelloBoard(val_x, val_y)

    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN1D()
    print(model)
    print(summary(model, (BATCH_SIZE, 64, 1)))

    OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LR)

    train_loss, train_acc, val_loss, val_acc = train(
        train_loader=train_dataloader, val_loader=val_dataloader, 
        model=model, batch_size=BATCH_SIZE, optimizer=OPTIMIZER,
        loss_func=OthelloLoss, acc_func=evaluate_accuracy, epoch=EPOCHS
    )