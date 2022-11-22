import os
import cv2
import pickle
import glob
import numpy as np
import random
import torch
import os
import copy
import re
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

class StockDataset(Dataset):
    def __init__(self, preprocessing_data, encode_size):
        self.encode_size = encode_size
        self.preprocessing_data = preprocessing_data   # only join two csv file together
        self.processing_data = self.text_processing()  # remove punc, clean data, stop words, etc
        self.vector_data = self.to_vector()
        self.data = None
        self.label = None

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        data, label = self.to_tensor(data, label)
        sample = (data, label)
        return sample

    @staticmethod
    def to_tensor(data, label):
        data = torch.from_numpy(data).float()
        label = torch.tensor(label)
        return data, label
    
    def text_processing(self):
        # all news
        headlines = list(self.preprocessing_data['News'])
        head = copy.deepcopy(headlines)
        headlines = []
        for para in head:
            para_re = self.clean_text(para)  # clean data first
            para_re = self.remove_stop(para_re)  # remove stop
            headlines.append(para_re)
        # join all words to a long sentence
        t_data = []
        for head in headlines:
            t_data.append(' '.join(d for d in head))
        return np.array(t_data)
    
    def to_vector(self):
        # 2-grams
        max_features = self.encode_size
        countvector = CountVectorizer(ngram_range=(1,2), max_features=max_features)
        vec_news = countvector.fit_transform(self.processing_data)
        return vec_news.toarray()

    def split_dataset(self):
        train_num = int(0.8 * len(self.processing_data))
        print('train_num: ', train_num)
        print('test_num: ', len(self.processing_data) - train_num)

        # Vectorized text as X
        X_train = self.vector_data[:train_num] 
        X_test = self.vector_data[train_num:] 

        # Close price as y
        y_train = self.preprocessing_data["Label"][:train_num].values
        y_test = self.preprocessing_data["Label"][train_num:].values
        return X_train, y_train, X_test, y_test
    
    @staticmethod
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
    
    @staticmethod
    def remove_stop(sentence):
        '''
        Utilzing word_tokenize to split words from a sentence and remove stop words.
        '''
        # collect stopwords from nltk
        stop = stopwords.words('english')
        stop += ['``', ':', '?', '.', '&', ';', '-', 't', 's', ',', '$', '|', "''" , '--', 'b']
        print("stop words: ", stop)
        word_tokens = word_tokenize(sentence)
        return [word for word in word_tokens if word not in stop]
