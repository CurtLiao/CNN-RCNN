
# coding: utf-8

# In[12]:


import numpy as np
np.random.seed(10)
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras import optimizers
import os
from PIL import Image
from keras.layers import Dropout
from matplotlib import pyplot as plt
import cv2
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


nameDatasets = "D:/testcnn/"
imgSize = (224,224)

dict_label = {"smoke":0,"cigar":1}
imgs = []
labels = []
labels_hot = []


# In[14]:


def show_train_history(train_history, train, validation):

    plt.plot(train_history.history[train])

    plt.plot(train_history.history[validation])

    plt.title('Train history')

    plt.ylabel(train)

    plt.xlabel('Epoch')

    legendLoc = 'lower right' if(train=='acc') else 'upper right'

    plt.legend(['train', 'validation'], loc=legendLoc)

    plt.show()


# In[15]:


def loadimg(folder):
    global imgs,labels,labels_hot,dict_label
    for filename in os.listdir(folder):
        label = os.path.basename(folder)
        className = np.asarray(label)
        if label is not None:
            labels.append(className)
            labels_hot.append(dict_label[label])
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            im2 = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
            imgs.append(np.array(im2))


# In[16]:


for folder in glob.glob(nameDatasets + "/*"):
    print(folder)
    loadimg(folder)

#將圖片轉為nmupy    
imgs = np.array(imgs)
labels_hot = np.array(labels_hot)
print("images.shape = {},labels_hot.shape=={}".format(imgs.shape,labels_hot.shape))

#查看一張圖片
sampleId = 80
plt.imshow(imgs[sampleId])
print("Label:{} , ID:{}, shape:{}".format(labels[sampleId],labels_hot[sampleId],imgs[sampleId].shape))


# In[17]:


(trainData, testData, trainLabels, testLabels) = train_test_split(imgs, labels_hot, test_size=0.2, random_state=42)


# In[18]:


print("trainData records: {}".format(len(trainData)))

print("testData records: {}".format(len(testData)))

print("trainData.shape={} trainLabels.shape={}".format(trainData.shape,trainLabels.shape))

print("testData.shape={} testLabels.shape={}".format(testData.shape,testLabels.shape))


# In[19]:


#label獨熱化編碼
trainLabels_hot = np_utils.to_categorical(trainLabels)

testLabels_hot = np_utils.to_categorical(testLabels)

#data normalize
trainData_normalize = trainData.astype('float32') / 255.0

testData_normalize = testData.astype('float32') / 255.0



# In[20]:


#model

model = Sequential()
#第一層捲積層
model.add(Conv2D(filters=16, kernel_size=(7, 7), padding='same', input_shape=(imgSize[0], imgSize[1], 3), activation='relu'))
#model.add(Dropout(rate=0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))


#第二層捲積層
model.add(Conv2D(filters=32, kernel_size=(7, 7), padding='same', input_shape=(imgSize[0], imgSize[1], 3), activation='relu'))
#model.add(Dropout(rate=0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))


#第三層捲積層
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same',input_shape=(imgSize[0], imgSize[1], 3), activation='relu'))
#model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))


#平坦
model.add(Flatten())
model.add(Dropout(rate=0.5))

#隱藏
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))

#全連結層
model.add(Dense(len(dict_label), activation='softmax'))

model.summary()


# In[21]:


#trainParas= optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])


train_history=model.fit(x=trainData_normalize, y=trainLabels_hot, validation_data=(testData_normalize, testLabels_hot), validation_split=0.2, epochs=15, batch_size=128, verbose=1)


# In[23]:


scores = model.evaluate(testData_normalize,testLabels_hot)
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))

show_train_history(train_history,'acc','val_acc')

show_train_history(train_history,'loss','val_loss')


# In[24]:


prediction = model.predict_classes(testData_normalize)
print(prediction)

pd.crosstab(testLabels, prediction, rownames=['label'], colnames=['predict'])

print(classification_report(testLabels, prediction))


# In[ ]:


img = cv2.imread("D:/testcnn/smoke/255.jpg")
# 画矩形框
cv2.rectangle(img, (10,10), (212,212), (0,255,0), 4)
# 标注文本
font= cv2.FONT_HERSHEY_SIMPLEX
text = '001'
cv2.putText(img, text, (180, 180), font,0.5, (0,0,255), 1)
cv2.imwrite('001_new.jpg', img)

