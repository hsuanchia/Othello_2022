import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from early_stop import early_stop
from tqdm import tqdm
import os,shutil
import matplotlib.pyplot as plt


### Config
CURRENT_PATH = os.path.dirname(__file__)
BEFORE_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.path.pardir)) 
s10 = f'{BEFORE_PATH}/train_data_0130/train_data_0130_s10_valid_100000.txt'
BOARD_SIZE = 8
MAXLEN = 128
BATCH_SIZE = 32
EPOCHS = 10000
LR = 1e-4
WEIGHT_SAVE_PATH = f'{CURRENT_PATH}/s10_allvalid_multilabel_weights/torch/'
EARLY_STOP = early_stop(save_path=f'{WEIGHT_SAVE_PATH}', mode='min', monitor='val_loss', patience=10)
try:
    shutil.rmtree(WEIGHT_SAVE_PATH)
except:
    pass
os.makedirs(WEIGHT_SAVE_PATH)

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

class OthelloLoss(nn.Module):
    def __init__(self):
        super(OthelloLoss, self).__init__()
        
    def forward(self, pred, label, th):
       
        sub_tmp = pred - label
        #print('pred',pred)
        #print('label',label)
        #print('sub_tmp',sub_tmp)
        sub_miss = torch.where((sub_tmp<(-0.5)),sub_tmp,0.0)
        sub_wrong = torch.where((sub_tmp>0.5),sub_tmp,0.0)
        sub_miss_wrong = torch.add(sub_miss,sub_wrong)
        #print('sub_miss',sub_miss)
        #print('sub_wrong',sub_wrong)
        #print(sub_miss_wrong)
        sub_result = torch.where((sub_miss_wrong<0.0), (1.0+sub_miss_wrong)*10000, sub_miss_wrong*10000)
        #print('sub_result',sub_result)
        sub_result = torch.sum(sub_result,dim=1)
        #print('sub_result',sub_result)
        sub_result = torch.mean(sub_result)
        #print(sub_result)
        return sub_result
    
def OthelloLoss_(pred, label, th):
    
    pred = torch.where(pred > th, True, False)
    label = torch.where(label == 1.0, True, False)

    #ex: p:110 g:011
    pred_gt_or = torch.bitwise_or(pred, label)                          #pred+gt where 1                     =>110 or 011 = 111
    denominator = torch.sum(torch.where(pred_gt_or == True, 1, 0),1)    #total pred+gt where 1 position      =>p or q:111 => 3

    pred_gt_xor = torch.bitwise_xor(pred, label)                        #not correct position                =>110 xor 011 = 101
    molecular =  torch.sum(torch.where(pred_gt_xor == True, 1, 0),1)    #total pred not correct num          =>p xor q:101 => 2 
    position_loss = molecular / denominator                             #position loss = (not correct num) / (total postion num)
    position_loss = torch.nan_to_num(position_loss, nan=0.0)
    position_batch_loss =  torch.div(torch.sum(position_loss), len(position_loss))

    wrong_and_xor = torch.bitwise_and(pred,pred_gt_xor)                     #wrong postion                       =>p and(p xor q) =  110 and 101 = 100
    wrong_num =  torch.sum(torch.where(wrong_and_xor == True, 1, 0),1)      #wrong num                           => p and(p xor q) = 100 => 1  
    wrong_penalty_loss = 0.5 * wrong_num
    wrong_penalty_batch_loss = torch.div(torch.sum(wrong_penalty_loss), len(wrong_penalty_loss))

    total_loss = position_batch_loss + wrong_penalty_batch_loss
    
    
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
            
            #criterion = loss_func()
            #loss = criterion(pred, train_y, th=0.5)
            loss = loss_func(pred, train_y)       

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

            #criterion = loss_func()
            #loss = criterion(pred, val_y, th=0.5)
            loss = loss_func(pred, val_y)
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

def plot_statistics(train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                    SAVE_MODELS_PATH):
    '''
    統計train、valid的loss、acc
    '''
    fig, ax = plt.subplots()
    epcoh = [x for x in range(len(train_loss))]
    ax2 = ax.twinx()
    t_loss = ax.plot(train_loss,color='green',label='train_loss')
    v_loss = ax.plot(valid_loss,color='red',label='valid_loss')
    t_acc = ax2.plot(train_acc,color='#00FF55',label='train_acc')
    v_acc = ax2.plot(valid_acc,color='#FF5500',label='valid_acc')
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel("acc")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig(f'{SAVE_MODELS_PATH}/train_statistics',bbox_inches='tight')
    plt.figure()

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
    loss_func = nn.BCELoss()
    print(model)
    print(summary(model, (BATCH_SIZE, 64, 1)))

    OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LR)

    train_loss, train_acc, val_loss, val_acc = train(
        train_loader=train_dataloader, val_loader=val_dataloader, 
        model=model, batch_size=BATCH_SIZE, optimizer=OPTIMIZER,
        loss_func=loss_func, acc_func=evaluate_accuracy, epoch=EPOCHS
    )

    plot_statistics(train_loss, train_acc, val_loss, val_acc, WEIGHT_SAVE_PATH)

