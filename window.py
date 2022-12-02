import os
from copy import deepcopy
import numpy as np
import pandas as pd
# import pandas_datareader as web
import datetime as dt
from interval import Interval
from collections import defaultdict
from itertools import permutations

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# hyper-params
max_features = 10000
batch_size = 64
learning_rate = 0.0001
n_epoch = 100
num_classes = 3  # three labels
window_size = 32
print_freq = 100
tolerance = 20
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# pytorch dataset

class StockDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        assert len(self.data) == len(self.label), "data and label should be same size at 0"
        return len(self.label)

    @staticmethod
    def to_tensor(data, label):
        data = torch.from_numpy(data.reshape((-1, data.shape[-1]))).float()
        label = torch.tensor(label)
        return data, label

    def __getitem__(self, index):
        data = self.data[index, :, :]
        label = self.label[index]
        data, label = self.to_tensor(data, label)
        sample = (data, label)
        return sample


# lstm attention model

class LSTM_attn(nn.Module):
    def __init__(self, in_f, out_f, hidden_size=512):
        super(LSTM_attn, self).__init__()
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(
            input_size=in_f,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True
        )
        self.f1 = nn.Linear(512, 64)
        self.f2 = nn.Linear(64, out_f)
        self.r = nn.ReLU()
        self.d = nn.Dropout(0.3)

    def attn(self, lstm_output, h_t):
        # lstm_output [bs, clips, hiden]  h_t[bs, hiden]
        h_t = h_t.unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, h_t) # lstm_output [bs, clips, hidden] ;h_t [bs, hidden, 1] --> attn [bs, clips, 1]
        attn_weights = attn_weights.squeeze(2)
        attention = F.softmax(attn_weights, dim = 0)
        # bmm : [bs, hidden, clips] [bs, clips, 1]
        attn_out = torch.bmm(lstm_output.transpose(1, 2), attention.unsqueeze(2)) # [bs, hidden, 1]
        return attn_out.squeeze(2) # [bs, hidden]

    def forward(self, x):
        bs = x.size()[0]
        window_size = x.size()[1]
        # x = x.view(bs // self.window_size, self.window_size, -1)
        self.LSTM.flatten_parameters()
        x, (hn,hc) = self.LSTM(x) # x.shape -> bs,clip,512
        x_last = x[:, -1, :] # x[:,-1,:].shape [bs, 512]

        if int(x.size()[1]) != 1:
            # attention
            x = self.attn(x, x_last)
            x = self.d(self.r(self.f1(x)))   
            x = self.f2(x) # [8, 128] --> [8, 2]
        else:
            # direct fc
            x = self.d(self.r(self.f1(x_last.reshape(-1, self.hidden_size))))
            x = self.f2(x)
        return x.view(bs, -1)  # expected output.shape --> [8, 2]


news = pd.read_csv('RedditNews.csv')
stock = pd.read_csv('stock.csv')


# Merge those rows with same Date time for news data
# Set Date as index

aggregation_functions = {'News': 'sum'}
news_new = news.groupby(news['Date']).aggregate(aggregation_functions)


# Set Date as index

stock.set_index('Date', inplace = True)

# Use "inner join" to merge news and stock data together

data = pd.concat([news_new, stock], axis=1, join = 'inner')
data.isnull().sum()
data.reset_index(inplace = True)


n = len(data)
train_data = data[data['Date'] < '20150101']
test_data = data[data['Date'] > '20141231']
print('train data:', len(train_data))
print('test data:', len(test_data))


ori_data = train_data['Close'].values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))
# scaler = PolynomialFeatures(degree=64, interaction_only=False, include_bias=False)
scaled_data = scaler.fit_transform(ori_data)
scaled_data.shape

ori_data_test = test_data['Close'].values.reshape(-1,1)
scaled_data_test = scaler.fit_transform(ori_data_test)

ori_data_all = data['Close'].values.reshape(-1,1)
scaled_data_all = scaler.fit_transform(ori_data_all)


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import copy
import re

# all news
headlines = list(data['News'])

# collect stopwords from nltk
stop = stopwords.words('english')
stop += ['``', ':', '?', '.', '&', ';', '-', 't', 's', ',', '$', '|', "''" , '--', 'b']
print("stop words: ", stop)

def clean_text(text):
    '''
    Make text lowercase, remove text in square brackets,remove links
    and remove words containing numbers.
    '''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("b'*", '', text)
    return text

def remove_stop(sentence):
    '''
    Utilzing word_tokenize to split words from a sentence and remove stop words.
    '''
    word_tokens = word_tokenize(sentence)
    return [word for word in word_tokens if word not in stop]

head = copy.deepcopy(headlines)
headlines = []
for para in head:
    para_re = clean_text(para)  # clean data first
    para_re = remove_stop(para_re)  # remove stop
    headlines.append(para_re)
    
# join all words to a long sentence
t_data = []
for head in headlines:
    t_data.append(' '.join(d for d in head))

print('Before: ', data['News'][0])
print(" ")
print('After: ', t_data[0])



# feature engineering
from sklearn.feature_extraction.text import CountVectorizer

# 2-grams
countvector=CountVectorizer(ngram_range=(1,2), max_features=128)

# transform input to numpy format
t_data = np.array(t_data)
vec_news = countvector.fit_transform(t_data)

# train and test data split
vec_news_train = vec_news[data['Date'] < '20150101'].toarray()
vec_news_test = vec_news[data['Date'] > '20141231'].toarray()


# windowsize adjust
combin = [1,3,5,7,10,15,20,25,30]
combinations = list(permutations(combin, 2))
combinations.extend(list(zip(combin, combin)))  # add diag element
print("Combination length:", len(combinations))

# cutoff setting
cutoff = Interval(-0.03, 0.03)

# metric dict define
metric_dict = defaultdict(tuple)
error_list = []

for index in range(len(combinations)):
    try:
        prediction_days, window_size = combinations[index]

        # gen label for both train and test
        ratio_list = []
        for x in range(prediction_days, len(scaled_data_all)-window_size):  
            ratio = (ori_data_all[x + window_size, :] - ori_data_all[x, :]) / (ori_data_all[x, :] + 0.00001)
            ratio_list.append(float(ratio))
        temp_ratio = np.array(sorted(ratio_list)) # store the sorted ratio
        # cutoff1 = np.percentile(temp_ratio, 33)
        # cutoff2 = np.percentile(temp_ratio, 66)
        cutoff = np.percentile(temp_ratio, 66) # choose the 66.7% quantile as cutoff
        # print("cutoff1:", cutoff1)
        # print("cutoff2:", cutoff2)
        print("cutoff:", cutoff)
        # new label
        # y_train = [
        #     0. 
        #     if float(td) < cutoff1 else 2. 
        #     if float(td) > cutoff2 else 1. 
        #     for td in temp_ratio
        # ]

        # train label
        ratio_list = []
        for x in range(prediction_days, len(scaled_data)-window_size):  
            ratio = (ori_data[x + window_size, :] - ori_data[x, :]) / (ori_data[x, :] + 0.00001)
            ratio_list.append(float(ratio))
        temp_ratio = np.array(ratio_list) # store the sorted ratio
        y_train = np.array([0. if float(td) < -cutoff else 2. if float(td) > cutoff else 1. for td in temp_ratio])

        # print the distribution of three labels
        print(f"Slow: {np.where(y_train==0.)[0].size}, Flat: {np.where(y_train==1.)[0].size}, Fast: {np.where(y_train==2.)[0].size}", )

        # train data
        x_train = []
        for x in range(prediction_days, len(scaled_data)-window_size):      ######
            x_feat = np.concatenate((vec_news_train[x-prediction_days:x, :], 
                                    scaled_data[x-prediction_days:x, :]), 1)
            x_train.append(x_feat)
            # regression label
            # y_train.append(scaled_data[x+window_size, :])                 ###### predict window_size days after
            # classification label

        # data gen
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], -1))

        # train data
        train_dataset = StockDataset(data=x_train, label=y_train)

        # train loader
        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
        )

        # model
        net = LSTM_attn(
                in_f=x_train.shape[-1], 
                out_f=num_classes, 
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
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, StepLR
        scheduler = (
                CosineAnnealingWarmRestarts(
                    optimizer, 
                    T_0=10, 
                    T_mult=5, 
                    eta_min=1e-6, 
                    last_epoch=-1)
        )

        # train
        total_loss, total_acc = 0., 0.
        loss_list, acc_list = [], []
        best_acc = 0.
        net.train()
        for epoch in range(n_epoch):
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                # print('output: ',outputs)
                # print('labels: ',labels)
                # loss = F.mse_loss(outputs, labels.float())
                loss = F.cross_entropy(outputs, labels.long())

                pred_classes = torch.max(outputs, dim=1)[1]
                correct = torch.eq(pred_classes, labels.to(device)).sum()
                acc = (correct / len(labels))
                total_acc += acc

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                loss_list.append(float(loss.detach().cpu().numpy()))
                acc_list.append(float(acc))
                
                if acc > best_acc:
                    best_model = deepcopy(net)

        # data gen
        actual_prices = test_data['Close'].values
        total_dataset = pd.concat((train_data['Close'], test_data['Close']), axis=0)
        model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values.reshape(-1,1)
        ori_model_inputs = deepcopy(model_inputs)
        model_inputs = scaler.transform(model_inputs)

        # test data
        ratio_list = []
        for x in range(prediction_days, len(scaled_data_test)-window_size):  
            ratio = (ori_data_test[x + window_size, :] - ori_data_test[x, :]) / (ori_data_test[x, :] + 0.00001)
            ratio_list.append(float(ratio))
        temp_ratio = np.array(ratio_list) # store the sorted ratio

        y_test = [0. if float(td) < -cutoff else 2. if float(td) > cutoff else 1. for td in temp_ratio]
        y_test = np.array(y_test)

        # print the distribution of three labels
        print(f"Slow: {np.where(y_test==0.)[0].size}, Flat: {np.where(y_test==1.)[0].size}, Fast: {np.where(y_test==2.)[0].size}", )


        # test
        x_test = []
        for x in range(prediction_days, len(scaled_data_test)-window_size):
            x_feat = np.concatenate((vec_news_test[x-prediction_days:x, :], 
                                    scaled_data_test[x-prediction_days:x, :]), 1)
            x_test.append(x_feat)

        # data gen
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], -1))

        # test data
        test_dataset = StockDataset(data=x_test, label=y_test.reshape((-1)))

        # test loader
        test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                drop_last=True,
        )

        # inference
        net.eval()
        actual, pred = [], []

        with torch.no_grad():
            for i, (test_inputs, test_labels) in enumerate(test_loader):
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)
                test_outputs = best_model(test_inputs)
                pred.extend(
                    torch.max(test_outputs, dim=1)[1]
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                    .tolist()
                )

                actual.extend(
                    test_labels
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                    .tolist()
                )

        y_true, y_pred = np.array(actual), np.array(pred)

        # calculate metrics
        acc_score = accuracy_score(y_true, y_pred)
        print(f"combinations_{index}: acc: {acc_score}")
        metric_dict[f'combinations_{index}'] = (y_true.tolist(), y_pred.tolist(), acc_score)
        
        # print(classification_report(y_true, y_pred))
        # print(accuracy_score(y_true, y_pred))
        # confusion_matrix(y_true, y_pred)

        # print(y_pred)
    
    except Exception as e:
        print(e)
        error_list.append(index)