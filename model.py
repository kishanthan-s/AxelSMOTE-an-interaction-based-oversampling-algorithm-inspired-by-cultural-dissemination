#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:22:30 2024

@author: aselahevapathige
"""



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import statistics
from sklearn import preprocessing
#from intricate_features import *
#from label_features import *
#from decomposition_features import *
import time
from AxelrodCulturalOversampler import AxelrodCulturalOversampler
#from AxelrodCulturalUndersampler import HashCulturalOversampler
from imblearn.combine import SMOTETomek


import os
os.environ["OMP_NUM_THREADS"] = '1'

import warnings
warnings.filterwarnings('ignore')




class Data(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(X_train.astype(np.float32).to_numpy())
        #self.y = torch.from_numpy(y_train.to_numpy()).type(torch.LongTensor)
        self.y = torch.tensor(pd.Categorical(y_train).codes, dtype=torch.long)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len
    
    
class MLPModel(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    
    def forward(self, x):
        af = F.relu
        x = af(self.linear1(x))
        x = self.linear2(x)
        return x 
    

'''
ILPD
annthyroid_21feat_normalised
kc1
'''

#dataset
df = pd.read_csv('data/glass.csv') #dataset name
df = df.fillna(0)
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]
X=(X-X.min())/(X.max()-X.min())




''' batch size
pendigits - 5000
diabetes - 500
page-blocks - 5000
glass - 500
wisconsin - 500
annthyroid_21feat_normalised -2500
bank-additional-full_normalised -2500
'''

batch_size = 2500
num_classes = len(Y.value_counts())

# number of features (len of X cols)
input_dim = X.shape[1]
# number of hidden layers
hidden_dim = 64
# number of classes (unique of y)
output_dim = num_classes
f1 = F1Score(num_classes=num_classes, task='multiclass')


precision_list = []
recall_list = []
f1_list = []
balanced_accuracy_list = []
gmean_list = []
roc_auc_list = []


start_time = time.time()
from imblearn.over_sampling import SMOTE
import pandas as pd
dataset_name="glass"
for kn in range(1, 7):
    for ct in [1, 2, 4, 8, 12, 16, 20]:
        for st in [0.0, 0.2, 0.4, 0.6, 0.8, 1]:
            for ir in [0.0, 0.2, 0.4, 0.6, 0.8, 1]:
                for seed in range(0, 10):
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed, stratify=Y)
                    
                    # Store original column names if X_train is a DataFrame
                    if hasattr(X_train, 'columns'):
                        column_names = X_train.columns
                    else:
                        column_names = None
                    
                    # Apply SMOTE oversampling
                    #smote = SMOTE(random_state=seed)
                    smote = AxelrodCulturalOversampler(random_state=seed, k_neighbors=kn, cultural_traits=ct, similarity_threshold=st, influence_rate=ir)
                    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
                    
                    # Convert back to pandas DataFrames/Series (which is what your Data class expects)
                    if column_names is not None:
                        X_train_resampled = pd.DataFrame(X_train_resampled, columns=column_names)
                    else:
                        X_train_resampled = pd.DataFrame(X_train_resampled)
                    
                    #Y_train_resampled = pd.Series(Y_train_resampled)
                    
                    # Continue with your original code
                    traindata = Data(X_train_resampled, Y_train_resampled)
                    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=0)
                    
                    clf = MLPModel(input_dim=X_train_resampled.shape[1], hidden_dim=hidden_dim)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(list(clf.parameters()), lr=0.05)
                    epochs = 200
                    
                    # Rest of your training loop remains the same
                    for epoch in range(epochs):
                        running_loss = 0.0
                        for i, data in enumerate(trainloader, 0):
                            inputs, labels = data
                            optimizer.zero_grad()
                            outputs = clf(inputs)
                            ce_loss = criterion(outputs, labels)
                            loss = ce_loss
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                        #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
                
                    testdata = Data(X_test, Y_test)
                    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=0)
                
                
                    dataiter = iter(testloader)
                    inputs, labels = next(dataiter)
                
                    outputs = clf(inputs)
                    __, predicted = torch.max(outputs, 1)
                
                    pred = []
                    lbl = []
                    batches = 0
                    # no need to calculate gradients during inference
                    with torch.no_grad():
                        for data in testloader:
                            batches = batches + 1
                            inputs, labels = data
                            # calculate output by running through the network
                            outputs = clf(inputs)
                            # get the predictions
                            __, predicted = torch.max(outputs.data, 1)
                            
                            # update results
                            pred.append(predicted)
                            lbl.append(labels)
                    
                    pred = torch.cat(pred, dim=0) 
                    lbl = torch.cat(lbl, dim=0) 
                    
                    # Calculate all metrics
                    f1 = 100 * f1_score(lbl, pred, average = 'macro')
                    precision = 100 * precision_score(lbl, pred, average = 'macro')
                    recall = 100 * recall_score(lbl, pred, average = 'macro')
                    balanced_accuracy = 100 * balanced_accuracy_score(lbl, pred)
                    
                    # Print results for current seed
                    #print('F1-Score of the network for seed ', seed, ' on the test data: ', f1)
                    #print('Precision of the network for seed ', seed, ' on the test data: ', precision)
                    #print('Recall of the network for seed ', seed, ' on the test data: ', recall)
                    #print('Balanced Accuracy of the network for seed ', seed, ' on the test data: ', balanced_accuracy)
                    #print('---')
                    
                    # Append to lists
                    f1_list.append(f1)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    balanced_accuracy_list.append(balanced_accuracy)
                
                
                # End time
                end_time = time.time()
                
                # Measure elapsed time in seconds
                elapsed_time = end_time - start_time
                
                # Convert to milliseconds
                elapsed_time_ms = elapsed_time * 1000
                
                
                
                total_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
                
                print(f"Total learnable parameters: {total_params}")
                print(f"Execution Time: {elapsed_time_ms:.3f} ms")
                # Print summary statistics for all metrics
                print('F1-Score Mean: ', statistics.mean(f1_list),  ', Std Deviation: ', statistics.stdev(f1_list))
                print('Precision Mean: ', statistics.mean(precision_list),  ', Std Deviation: ', statistics.stdev(precision_list))
                print('Recall Mean: ', statistics.mean(recall_list),  ', Std Deviation: ', statistics.stdev(recall_list))
                print('Balanced Accuracy Mean: ', statistics.mean(balanced_accuracy_list),  ', Std Deviation: ', statistics.stdev(balanced_accuracy_list))
                
                os.makedirs("results", exist_ok=True)
            
                filename = f"results/summary_{dataset_name}_kn_{kn}_ct_{ct}_st_{st}_ir_{ir}.txt" # You can change this to any valid filename 
            
                with open(filename, "w") as f:
                    f.write(f"Execution Time: {elapsed_time_ms:.3f} ms\n")
            
                    f.write(f"Total learnable parameters: {total_params}\n")
            
                    f.write(f"F1-Score Mean: {statistics.mean(f1_list):.4f}, Std Deviation: {statistics.stdev(f1_list):.4f}\n")
                    f.write(f"Precision Mean: {statistics.mean(precision_list):.4f}, Std Deviation: {statistics.stdev(precision_list):.4f}\n")
                    f.write(f"Recall Mean: {statistics.mean(recall_list):.4f}, Std Deviation: {statistics.stdev(recall_list):.4f}\n")
                    f.write(f"Balanced Accuracy Mean: {statistics.mean(balanced_accuracy_list):.4f}, Std Deviation: {statistics.stdev(balanced_accuracy_list):.4f}\n")
                       