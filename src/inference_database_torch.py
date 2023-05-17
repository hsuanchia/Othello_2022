import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from early_stop import early_stop
from tqdm import tqdm
import os,shutil
import matplotlib.pyplot as plt

### Configs
BOARD_SIZE = 8
BATCH_SIZE = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

def build_test_data_2d(path):
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
        l_tmp = l_tmp[:-1]
        # print(l_tmp)
        for y in l_tmp:
            label[int(y[1]) * BOARD_SIZE + int(y[3])] = 1
        board.append(cur_board)
        labels.append(label)
    return board, labels

def build_test_data_1d(path):
    board, labels = [], []
    f = open(path)
    for l in f.readlines():
        cur_board = []
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
                    cur_board.append(1)
                if(j[0] == '0'):
                    cur_board.append(0)
                if(j[0] == '-'):
                    cur_board.append(-1)
        l_tmp = l_tmp[:-1]
        for y in l_tmp:
            label[int(y[1]) * BOARD_SIZE + int(y[3])] = 1
        board.append(cur_board)
        labels.append(label)
    
    return board, labels

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
        # x = self.conv2(x)
        # x = nn.functional.relu(x)
        # x = self.maxpool(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = self.linear3(x)
        x = nn.functional.sigmoid(x)

        return x

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

### Inference stage
def inference(data_loader, model):
    test_avg_acc = 0
    total_acc = 0
    model.eval()    #預測要把model變成eval狀態
    with torch.no_grad():
        for test_x, test_y in tqdm(data_loader):
            test_x = test_x.to(device, dtype=torch.float)
            test_y = test_y.to(device, dtype=torch.float)

            pred = model(test_x)

            ### Evaluate accuracy
            # pred = pred.detach().cpu().numpy().tolist()
            test_y = test_y.detach().cpu().numpy().tolist()
            hit = 0
            ans = pred.argmax(dim=1)
            for i in range(len(ans)):
                if test_y[i][int(ans[i])] == 1.0:
                        hit += 1
                # hit += acc_func(pred[i], val_y[i])
            batch_acc = hit / len(pred)
            total_acc += batch_acc

        test_avg_acc = round(total_acc/len(data_loader),4)        

    return test_avg_acc

if __name__ == '__main__':
    model = CNN1D()
    model.load_state_dict(torch.load('E:/hsuanchia_e/Othello_2022/src/s10_allvalid_multilabel_weights/torch/epoch_261_trainLoss_0.0453_trainAcc_60.19_valLoss_0.055_valAcc_56.7.pth'))
    model.to(device)
    test_path = './train_data_0130/test_data_0214_s10_valid_100001.txt'

    hit, total = 0, 0
    get_ans = True
    test_x, test_y = build_test_data_1d(test_path)
    print(f'Data length: {len(test_x)}')

    test_ds = OthelloBoard(test_x, test_y)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    test_acc = inference(test_loader, model)

    print(f'Test accuracy: {test_acc}')

