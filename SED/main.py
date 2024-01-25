# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:28:19 2024

@author: Snehit
"""


## Imports
import pandas as pd
import numpy as np
import os
from tabulate import tabulate
from iteration_utilities import flatten
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from STeLiN_US_Data import STeLiN_US_Events
from Model import CNN



if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')


''' User Inputs '''
disc_STeLiN_US = 'User needs to give address for the STeLiN-US dataset location on the local'
print_df = False   # print df if require

''' Read Event Meta Data '''
df = pd.read_pickle('df_polyphonic_meta.pkl')
if print_df==True:
    df.info() 


## Default Labels
Event_labels = list(set(list(flatten(df['Events'].tolist()))))
Event_labels.sort()

''' Load data '''
print('\n--- Extracting features and labels ---\n')
Data = STeLiN_US_Events(disc_STeLiN_US, df, 'Wavefile', 'Clip', 'Events', Event_labels)
print('\n--- Finished: Extracting features and labels ---\n')

## Train and Test Split (70% Train and 30% Test)
Len_Train = int(len(Data)*0.7)
Len_Valid = len(Data) - Len_Train

train, valid = torch.utils.data.random_split(Data, [Len_Train, Len_Valid])

## Dataloader for train and test
train_loader = DataLoader(train, batch_size=50, shuffle=True)
valid_loader = DataLoader(valid, batch_size=50, shuffle=True)




# Load CNN model to train
model = CNN(input_shape=(1,128,157), output_shape=len(Event_labels)).to(device)

## set learning rate
def setlr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return optimizer

def lr_decay(optimizer, epoch):
  if epoch%5==0:
    new_lr = learning_rate / (5**(epoch//5))
    optimizer = setlr(optimizer, new_lr)
    print(f'Changed learning rate to {new_lr}')
  return optimizer


## Parameters
learning_rate = 2e-4
epochs = 30
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
valid_losses = []
Accuracy = []
Recall_Score = []


'''Trainning Loop'''
def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, Event_labels, change_lr=None):
  for epoch in tqdm(range(1,epochs+1)):
    model.train()
    batch_losses=[]
    if change_lr:
      optimizer = change_lr(optimizer, epoch)
    for i, data in enumerate(train_loader):
      x, y = data
      optimizer.zero_grad()
      x = x.to(device, dtype=torch.float32)
      y = y.to(device, dtype=torch.float32)
      y_hat = model(x)
      
      loss = loss_fn(y_hat, y)
      loss.backward()
      batch_losses.append(loss.item())
      optimizer.step()
    train_losses.append(batch_losses)
    model.eval()
    batch_losses=[]
    trace_y = []
    trace_yhat = []
    for i, data in enumerate(valid_loader):
      x, y = data
      x = x.to(device, dtype=torch.float32)
      y = y.to(device, dtype=torch.float32)
      y_hat = model(x)
      loss = loss_fn(y_hat, y)
      trace_y.append(y.cpu().detach().numpy())
      look = y_hat
      look[look>=0.5], look[look<0.5] = 1, 0
      trace_yhat.append(look.cpu().detach().numpy())
      batch_losses.append(loss.item())
    valid_losses.append(batch_losses)
    trace_y = np.concatenate(trace_y)
    trace_yhat = np.concatenate(trace_yhat)
    Accuracy.append(accuracy_score(trace_y, trace_yhat))
    
    
    Recall_Score.append(recall_score(trace_y, trace_yhat, average='macro'))
    
    
    print(f'\nEpoch - {epoch}')
    
    print(tabulate([['Train-Loss', f"{np.mean(train_losses[-1]):0,.4f}"], 
                    ['Valid-Loss', f"{np.mean(valid_losses[-1]):0,.4f}"],
                    ['Valid-Accuracy', f"{Accuracy[-1]:0,.4f}"],
                    ['Recall-Score', f"{Recall_Score[-1]:0,.4f}"]], 
                    headers=['Metrics', 'Results'], tablefmt="grid"))
    
    if epoch==epochs:
        print(classification_report(trace_y, trace_yhat, target_names=Event_labels))


''' Train model'''
print('\n--- Training CNN model ---\n')
train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, Event_labels, lr_decay)
print('\n--- Finished: Training CNN model ---\n')


'''Display Results'''
train_losses = np.asarray(train_losses).ravel()
valid_losses = np.asarray(valid_losses).ravel()

plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.plot(train_losses, 'tab:blue')
plt.legend(['Train Loss'], fontsize='18')

plt.subplot(2,1,2)
plt.plot(valid_losses, 'tab:orange')
plt.legend(['Valid Loss'], fontsize='18')


plt.figure(figsize=(12,6))
plt.plot(Recall_Score, 'blue')
plt.plot(Accuracy, 'tab:green')
plt.xlabel('Epochs')
plt.legend(['Recall Score', 'Accuracy'], fontsize='18')
plt.title('Metrics')
