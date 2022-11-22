import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, StepLR

from dataset import StockDataset
from model import LSTM_attn
from early_stop import EarlyStopping
from logger import prepare_logger

def data_preprocessing(news, stock):
    # Merge those rows with same Date time for news data
    # Set Date as index
    aggregation_functions = {'News': 'sum'}
    news_new = news.groupby(news['Date']).aggregate(aggregation_functions)
    # Set Date as index
    stock.set_index('Date', inplace = True)
    # Use "inner join" to merge news and stock data together
    result = pd.concat([news_new, stock], axis=1, join = 'inner')
    result.isnull().sum()
    result.reset_index(inplace = True)
    # day1's price- day0's price= up/down trend of day1's price, we use day1's news to predict the close price trend of day1
    result['Label'] = np.where(result['Close'].shift(-1) - result['Close'] < 0, 0, 1)
    return result

# hyper-params
max_features = 32
batch_size = 64
learning_rate = 0.0001
n_epoch = 100
num_classes = 2
window_size = 32
print_freq = 100
tolerance = 20
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# read data
news = pd.read_csv('RedditNews.csv')
stock = pd.read_csv('stock.csv')

# preprocessing
result = data_preprocessing(news, stock)

# torch dataset
train_dataset = StockDataset(
        preprocessing_data=result, 
        encode_size=max_features, 
)
X_train, y_train, X_test, y_test = train_dataset.split_dataset()
train_dataset.data = X_train
train_dataset.label = y_train
test_dataset = copy.deepcopy(train_dataset)
test_dataset.data = X_test
test_dataset.label = y_test

# loader
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
)

test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
)

# model
net = LSTM_attn(
        in_f=max_features, 
        out_f=num_classes, 
        window_size=window_size
)
net = net.to(device)

# opt
optimizer = (
        torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr = learning_rate, 
            betas=(0.9, 0.999))
)

# sche
scheduler = (
        CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=5, 
            eta_min=1e-6, 
            last_epoch=-1)
)

# criterion
loss_func = nn.CrossEntropyLoss()

# initialize the early_stopping object
early_stopping = EarlyStopping(patience=tolerance)

# create logger
log = prepare_logger(log_dir='./', log_filename="log.txt", stdout=False)

total_loss, total_acc, best_acc = 0, 0, 0
net.train()
for epoch in tqdm(range(n_epoch)):
    total_loss, total_acc = 0, 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        # label is the last day of continous window_size days
        labels = labels.view(-1, window_size)[:, -1]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred_classes = torch.max(outputs, dim=1)[1]
        correct = torch.eq(pred_classes, labels.to(device)).sum()
        total_acc += (correct / (batch_size//window_size))
        total_loss += loss.item()
        
        if i % print_freq == 0:
            (log.info('Training --> [ Epoch: {} ] loss:{:.5f} acc:{:.5f} '
                .format(epoch+1, loss.item(), 
                correct*100/(batch_size//window_size)),)
            )
    
    net.eval()
    with torch.no_grad():
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            # label is the last day of continous window_size days
            labels = labels.view(-1, window_size)[:, -1]
            labels = labels.to(device)

            outputs = net(inputs)
            loss = loss_func(outputs, labels) 
            pred_classes = torch.max(outputs, dim=1)[1]
            correct = torch.eq(pred_classes, labels.to(device)).sum()
            acc = (correct / (batch_size//window_size))
            total_acc += acc
            total_loss += loss.item()

            if acc > best_acc:
                best_acc = acc
                log.info(f"Current best acc: {best_acc}")

            if i % print_freq == 0:
                (log.info('Testing --> [ Epoch: {} ] loss:{:.5f} acc:{:.5f} '
                    .format(epoch+1, loss.item(), 
                    correct*100/(batch_size//window_size)),)
                )

    # early stop
    early_stopping(total_loss, net)
    if early_stopping.early_stop:
        log.info("Early stopping")
        break

    net.train()
    