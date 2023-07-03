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
import seaborn as sns
import numpy as np
import gc


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


data_train = pd.read_csv("/kaggle/input/deep-halucination/train.csv")
data_val = pd.read_csv("/kaggle/input/deep-halucination/val.csv")
data_test = pd.read_csv("/kaggle/input/deep-halucination/test.csv")


# In[4]:


DATA_MEAN = (0.5, 0.5, 0.5)
DATA_STD = (0.5, 0.5, 0.5)


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


# In[7]:


class SetDate(Dataset):
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


dataset_train = SetDate(data_train, "/kaggle/input/deep-halucination/train_images/", train_transform)
train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True)
dataset_val = SetDate(data_val, "/kaggle/input/deep-halucination/val_images/", val_transform)
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


class NN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(12288, 128)
        self.classifier = nn.Linear(128, 96)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x    


# In[12]:


model = NN1().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_method(model, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[11]:


class NN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(12288, 128)
        self.classifier = nn.Linear(128, 96)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.classifier(x)
        return x  


# In[12]:


model2 = NN2().to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
train_method(model2, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[13]:


class NN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(12288, 256)
        self.classifier = nn.Linear(256, 96)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x      


# In[14]:


model3 = NN3().to(device)
optimizer = torch.optim.Adam(model3.parameters(), lr=0.001)
train_method(model3, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[15]:


class NN4(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(12288, 256)
        self.classifier = nn.Linear(256, 96)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.classifier(x)
        return x      


# In[ ]:


model4 = NN4().to(device)
optimizer = torch.optim.Adam(model4.parameters(), lr=0.001)
train_method(model4, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class NN5(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(12288, 128)
        self.fc_2 = nn.Linear(128, 256)
        self.classifier = nn.Linear(256, 96)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x    


# In[ ]:


model5 = NN5().to(device)
optimizer = torch.optim.Adam(model5.parameters(), lr=0.001)
train_method(model5, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class NN6(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(12288, 128)
        self.fc_2 = nn.Linear(128, 256)
        self.classifier = nn.Linear(256, 96)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_1(x)
        x = F.leaky_relu(x)
        x = self.fc_2(x)
        x = F.leaky_relu(x)
        x = self.classifier(x)
        return x     


# In[ ]:


model6 = NN6().to(device)
optimizer = torch.optim.Adam(model6.parameters(), lr=0.001)
train_method(model6, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class NN7(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(12288, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc_2 = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, 96)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x    


# In[ ]:


model7 = NN7().to(device)
optimizer = torch.optim.Adam(model7.parameters(), lr=0.001)
train_method(model7, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class NN8(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(12288, 256)
        self.dropout = nn.Dropout(0.25)
        self.fc_2 = nn.Linear(256, 256)
        self.classifier = nn.Linear(256, 96)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x    


# In[ ]:


model8 = NN8().to(device)
optimizer = torch.optim.Adam(model8.parameters(), lr=0.0015)
train_method(model8, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:




