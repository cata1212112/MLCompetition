#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


data_train = pd.read_csv("/kaggle/input/deep-halucination/train.csv")
data_val = pd.read_csv("/kaggle/input/deep-halucination/val.csv")
data_test = pd.read_csv("/kaggle/input/deep-halucination/test.csv")


# In[ ]:


DATA_MEAN = (0.5, 0.5, 0.5)
DATA_STD = (0.5, 0.5, 0.5)


# In[ ]:


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATA_MEAN, std=DATA_STD)
])


# In[ ]:


val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=DATA_MEAN, std=DATA_STD)
])


# In[ ]:


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


# In[ ]:


dataset_train = CustomDataset(data_train, "/kaggle/input/deep-halucination/train_images/", train_transform)
train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True)
dataset_val = CustomDataset(data_val, "/kaggle/input/deep-halucination/val_images/", val_transform)
val_loader = DataLoader(dataset_val, batch_size=128, shuffle=True)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


class CNN1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.bnom_1 = nn.BatchNorm2d(64)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(13*13*128, 128)
        self.fc_bnom_1 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bnom_1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
            
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_bnom_1(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x


# In[ ]:


model1 = CNN1().to(device)
optimizer = torch.optim.Adam(model1.parameters(), lr=0.0015)
train_method(model1, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class CNN2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=1)
        self.bnom_1 = nn.BatchNorm2d(64)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(13*13*128, 128)
        self.fc_bnom_1 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bnom_1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
            
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_bnom_1(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x


# In[ ]:


model2 = CNN2().to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.0015)
train_method(model2, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class CNN3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.bnom_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.bnom_2 = nn.BatchNorm2d(64)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(13*13*128, 128)
        self.fc_bnom_1 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bnom_1(x)
        x = self.conv_2(x)
        x = self.bnom_2(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
            
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_bnom_1(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x


# In[ ]:


model3 = CNN3().to(device)
optimizer = torch.optim.Adam(model3.parameters(), lr=0.0015)
train_method(model3, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class CNN4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=1)
        self.bnom_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
        self.bnom_2 = nn.BatchNorm2d(64)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(13*13*128, 128)
        self.fc_bnom_1 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bnom_1(x)
        x = self.conv_2(x)
        x = self.bnom_2(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
            
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_bnom_1(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x


# In[ ]:


model4 = CNN4().to(device)
optimizer = torch.optim.Adam(model4.parameters(), lr=0.0015)
train_method(model4, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class CNN5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.bnom_1 = nn.BatchNorm2d(64)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv_2 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.bnom_2 = nn.BatchNorm2d(64)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(13*13*128, 128)
        self.fc_bnom_1 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bnom_1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        
        x = self.conv_2(x)
        x = self.bnom_2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
            
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_bnom_1(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x


# In[ ]:


model5 = CNN5().to(device)
optimizer = torch.optim.Adam(model5.parameters(), lr=0.0015)
train_method(model5, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class CNN6(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=11, stride=2)
        self.bnom_1 = nn.BatchNorm2d(64)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.conv_2 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
        self.bnom_2 = nn.BatchNorm2d(64)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=1
                                      
        self.conv_3 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.bnom_3 = nn.BatchNorm2d(64)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(13*13*128, 128)
        self.fc_bnom_1 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bnom_1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
                                      
        x = self.conv_2(x)
        x = self.bnom_2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
                                      
        x = self.conv_3(x)
        x = self.bnom_3(x)
        x = F.relu(x)
        x = self.maxpool_3(x)
            
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_bnom_1(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x


# In[ ]:


model6 = CNN6().to(device)
optimizer = torch.optim.Adam(model6.parameters(), lr=0.0015)
train_method(model6, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class CNN7(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.bnom_1 = nn.BatchNorm2d(64)
        self.conv_11 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bnom_11 = nn.BatchNorm2d(64)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bnom_2 = nn.BatchNorm2d(128)
        self.conv_22 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.bnom_22 = nn.BatchNorm2d(128)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
                                     
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bnom_3 = nn.BatchNorm2d(256)
        self.conv_33 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bnom_33 = nn.BatchNorm2d(256)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=1)   
        
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(13*13*128, 128)
        self.fc_bnom_1 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bnom_1(x)
        x = F.relu(x)
        x = self.conv_11(x)
        x = self.bnom_11(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        
        x = self.conv_2(x)
        x = self.bnom_2(x)
        x = F.relu(x)
        x = self.conv_22(x)
        x = self.bnom_22(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
        
        x = self.conv_3(x)
        x = self.bnom_3(x)
        x = F.relu(x)
        x = self.conv_33(x)
        x = self.bnom_33(x)
        x = F.relu(x)
        x = self.maxpool_3(x)
        
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_bnom_1(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x


# In[ ]:


model7 = CNN5().to(device)
optimizer = torch.optim.Adam(model7.parameters(), lr=0.0015)
train_method(model7, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)


# In[ ]:


class CNN8(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.bnom_1 = nn.BatchNorm2d(64)
        self.conv_11 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bnom_11 = nn.BatchNorm2d(64)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bnom_2 = nn.BatchNorm2d(128)
        self.conv_22 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.bnom_22 = nn.BatchNorm2d(128)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
                                     
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bnom_3 = nn.BatchNorm2d(256)
        self.conv_33 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bnom_33 = nn.BatchNorm2d(256)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=1)   
        
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(13*13*128, 128)
        self.fc_bnom_1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bnom_1(x)
        x = F.relu(x)
        x = self.conv_11(x)
        x = self.bnom_11(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        
        x = self.conv_2(x)
        x = self.bnom_2(x)
        x = F.relu(x)
        x = self.conv_22(x)
        x = self.bnom_22(x)
        x = F.relu(x)
        x = self.maxpool_2(x)
        
        x = self.conv_3(x)
        x = self.bnom_3(x)
        x = F.relu(x)
        x = self.conv_33(x)
        x = self.bnom_33(x)
        x = F.relu(x)
        x = self.maxpool_3(x)
        
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_bnom_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# In[ ]:


model8 = CNN8().to(device)
optimizer = torch.optim.Adam(model8.parameters(), lr=0.0015)
train_method(model8, train_loader, optimizer, nn.CrossEntropyLoss(), 20, val_loader)

