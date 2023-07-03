#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from scipy.ndimage.interpolation import shift
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import parallel_backend
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data_train = pd.read_csv("data/train.csv")
data_val = pd.read_csv("data/val.csv")
data_test = pd.read_csv("data/test.csv")
images_train = data_train['Image']
images_val = data_val['Image']
y_val = np.array(data_val['Class'])
y_train = np.array(data_train['Class'])
images_test = data_test['Image']


# In[6]:


def load_images(image_list, path):
    current_images = []
    for img in image_list:
        img_aux = Image.open(path + "/" + img)
        current_images.append(np.asarray(img_aux).reshape(-1))
    return np.array(current_images)


# In[7]:


X_train = load_images(images_train, "data/train_images")
X_val = load_images(images_val, "data/val_images")
X_test = load_images(images_test, "data/test_images")


# In[ ]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
for n in [1, 3, 5, 7, 11, 25, 55, 155, 355, 555]:
    for w in ['uniform', 'distance']:
        for p in [1, 2]:
            kNN = KNeighborsClassifier(n_neighbors=n, weights=w, p=p, n_jobs=-1)
            with parallel_backend('threading', n_jobs=4):
                kNN.fit(X_train_scaled, y_train)
            pred = kNN.predict(X_val_scaled)
            print("n: {}, w: {}, p: {} am obtinut acuratetea: {}".format(n, w, p, accuracy_score(pred, y_val)))
        for m in ['cosine']:
            kNN = KNeighborsClassifier(n_neighbors=n, weights=w, metric=m, n_jobs=-1)
            with parallel_backend('threading', n_jobs=4):
                kNN.fit(X_train_scaled, y_train)
            pred = kNN.predict(X_val_scaled)
            print("n: {}, w: {}, m: {} am obtinut acuratetea: {}".format(n, w, m, accuracy_score(pred, y_val)))


# In[8]:


def values_to_bins(x, bins):
    x = np.digitize(x, bins)
    return x - 1


# In[9]:


def show_image(img):
    Image.fromarray(img).show()


# In[10]:


for bins in [3, 5, 7, 15, 25, 35, 45, 85, 125, 175, 200, 225, 255]:
    bn = np.linspace(0, 255, num=bins)
    X_train_bin = values_to_bins(X_train, bn)
    X_val_bin = values_to_bins(X_val, bn)
    for a in [0, 0.1, 0.01, 0.001, 0.5, 1]:
        mnb = MultinomialNB(alpha=a, force_alpha=True)
        mnb.fit(X_train_bin, y_train)
        pred = mnb.predict(X_val_bin)
        print("bins: {}, a: {} am obtinut acuratetea: {}".format(bins, a, accuracy_score(pred, y_val)))


# In[17]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
kNN = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='cosine', n_jobs=-1)
kNN.fit(X_train_scaled, y_train)
pred = kNN.predict(X_val_scaled)

cm = confusion_matrix(y_val, pred)
precision = precision_score(y_val, pred, average=None)
recall = recall_score(y_val, pred, average=None)


# In[22]:


classes = [i for i in range(96)]
plt.bar(range(len(classes)), precision, color='red')
plt.xticks(np.arange(0, len(classes), 15), np.arange(0, len(classes), 15))
plt.ylabel('Precision')
plt.title('Precision')
plt.show()


# In[23]:


classes = [i for i in range(96)]
plt.bar(range(len(classes)), recall, color='red')
plt.xticks(np.arange(0, len(classes), 15), np.arange(0, len(classes), 15))
plt.ylabel('Recall')
plt.title('Recall')
plt.show()


# In[26]:


sns.heatmap(cm)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[27]:


bn = np.linspace(0, 255, num=25)
X_train_bin = values_to_bins(X_train, bn)
X_val_bin = values_to_bins(X_val, bn)
mnb = MultinomialNB(alpha=0.1, force_alpha=True)
mnb.fit(X_train_bin, y_train)
pred = mnb.predict(X_val_bin)


# In[28]:


cm = confusion_matrix(y_val, pred)
precision = precision_score(y_val, pred, average=None)
recall = recall_score(y_val, pred, average=None)


# In[29]:


classes = [i for i in range(96)]
plt.bar(range(len(classes)), precision, color='red')
plt.xticks(np.arange(0, len(classes), 15), np.arange(0, len(classes), 15))
plt.ylabel('Precision')
plt.title('Precision')
plt.show()


# In[30]:


classes = [i for i in range(96)]
plt.bar(range(len(classes)), recall, color='red')
plt.xticks(np.arange(0, len(classes), 15), np.arange(0, len(classes), 15))
plt.ylabel('Recall')
plt.title('Recall')
plt.show()


# In[31]:


sns.heatmap(cm)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




