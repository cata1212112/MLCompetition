#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gc


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


data_train = pd.read_csv("/kaggle/input/setuldedate/data/train.csv")
data_val = pd.read_csv("/kaggle/input/setuldedate/data/val.csv")
data_test = pd.read_csv("/kaggle/input/setuldedate/data/test.csv")


# In[4]:


DATA_MEAN = (0.5, 0.5, 0.5)
DATA_STD = (0.5, 0.5, 0.5)


# 

# In[5]:


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATA_MEAN, std=DATA_STD)
])


# In[6]:


val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=DATA_MEAN, std=DATA_STD)
])


# In[ ]:





# In[7]:


class CustomDataset(Dataset):
    def __init__(self, dataframe, path, transform=None):
        self.data = dataframe
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index]['Image']
        img = Image.open(self.path+img_path).convert('RGB')
        img = self.transform(img)
        label = self.data.iloc[index]['Class']
        return img, label


# In[8]:


dataset_train = CustomDataset(data_train, "/kaggle/input/setuldedate/data/train_images/", train_transform)
train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True)
dataset_val = CustomDataset(data_val, "/kaggle/input/setuldedate/data/val_images/", val_transform)
val_loader = DataLoader(dataset_val, batch_size=128, shuffle=True)


# In[9]:


def train_method(model, dataloader, optimizer, loss_fn, num_epochs, val_loader):
    model.train(True)
    loss_i_iter = 0
    for i in range(num_epochs):
        for batch, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            loss_i_iter = loss.item()
        val_stats = test_method(val_loader, model, loss_fn)
        correct = 0.
        for img_batch, y_batch in dataloader:
            img_batch = img_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(img_batch)
            correct += (pred.argmax(1) == y_batch).type(torch.float).sum().item()
        correct /= len(dataloader.dataset)
        print(f"Epoch {i}/{num_epochs} train loss: {loss_i_iter}, accuracy: {(100*correct):>0.1f}%, Validation: " + val_stats)


# In[10]:


def test_method(dataloader, model, loss_fn):
    correct = 0.
    val_loss = 0.
    size = len(dataloader.dataset)
    model.eval()
    with torch.no_grad():
        for img_batch, y_batch in dataloader:
            img_batch = img_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(img_batch)
            val_loss += loss_fn(pred, y_batch).item()
            correct += (pred.argmax(1) == y_batch).type(torch.float).sum().item()
    
    correct /= size
    return f"Accuracy: {(100*correct):>0.1f}%, Loss: {val_loss:>8f} \n"


# In[11]:


class ResidualBlock(nn.Module):
    def __init__(self, inch, outch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(inch, outch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outch)
        self.conv2 = nn.Conv2d(outch, outch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outch)
        self.relu = nn.ReLU()
        
        if stride != 1 or inch != outch:
            self.skip = nn.Sequential(
                nn.Conv2d(inch, outch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outch)
            )
        else:
            self.skip = nn.Identity()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.skip(residual)
        out = self.relu(out)
        return out


# In[12]:


class CNNRes1(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_block1 = ResidualBlock(3, 64, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(63 * 63 * 64, 128)
        self.fc_bnorm1 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, 96)
        
    def forward(self, x):
        
        out = self.residual_block1(x)
        out = self.maxpool1(out)
        
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc_bnorm1(out)
        out = F.relu(out)
        out = self.classifier(out)
        
        return out


# In[13]:


model1 = CNNRes1().to(device)
optimizer = torch.optim.Adam(model1.parameters(), lr=0.0015)
train_method(model1, train_loader, optimizer, nn.CrossEntropyLoss(), 10, val_loader)


# In[19]:


class CNNRes2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.residual_block1 = ResidualBlock(64, 64, 1)
        self.residual_block2 = ResidualBlock(64, 64, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.residual_block3 = ResidualBlock(64, 128, stride=2)
        self.residual_block4 = ResidualBlock(128, 128, 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15 * 15 * 128, 256)
        self.fc_bnorm1 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, 96)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.maxpool1(out)
        
        out = self.residual_block3(out)
        out = self.residual_block4(out)
        out = self.maxpool2(out)
        
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc_bnorm1(out)
        out = F.relu(out)
        out = self.classifier(out)
        
        return out


# In[20]:


model2 = CNNRes2().to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.0015)
train_method(model2, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[12]:


class CNNRes3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.residual_block1 = ResidualBlock(64, 64, 1)
        self.residual_block2 = ResidualBlock(64, 64, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.residual_block3 = ResidualBlock(64, 128, stride=2)
        self.residual_block4 = ResidualBlock(128, 256, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15 * 15 * 256, 96)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.maxpool1(out)
        
        out = self.residual_block3(out)
        out = self.residual_block4(out)
        out = self.maxpool2(out)
        
        out = self.flatten(out)
        out = self.fc1(out)
        
        return out


# In[13]:


model3 = CNNRes3().to(device)
optimizer = torch.optim.Adam(model3.parameters(), lr=0.0015)
train_method(model3, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class CNNRes4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.residual_block1 = ResidualBlock(64, 64, 1)
        self.residual_block2 = ResidualBlock(64, 64, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.residual_block3 = ResidualBlock(64, 128, stride=2)
        self.residual_block4 = ResidualBlock(128, 128, 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.residual_block5 = ResidualBlock(128, 256, stride=2)
        self.residual_block6 = ResidualBlock(256, 256, 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14 * 14 * 256, 256)
        self.fc_bnorm1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, 96)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.maxpool1(out)
        
        out = self.residual_block3(out)
        out = self.residual_block4(out)
        out = self.maxpool2(out)
        
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.maxpool3(out)
        
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc_bnorm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out


# In[ ]:


model4 = CNNRes4().to(device)
optimizer = torch.optim.Adam(model4.parameters(), lr=0.001)
train_method(model4, train_loader, optimizer, nn.CrossEntropyLoss(), 100, val_loader)

