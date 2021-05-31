# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:12:00 2021

@author: Dyari M.shareef
"""
#### Extracting MFCC's For every audio file
import pandas as pd
import librosa
import numpy as np
import os

metadata=pd.read_csv("C:\\Users\\Dyari M.shareef\\Desktop\\Audio-Auth\\train_test_split.csv")


def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

data=pd.concat((metadata['train_cat'][0:164], metadata['train_dog'][0:112]),axis=0)
yData=(['cat']*164) + (['dog']*112)


### Now we iterate through every audio file and extract features 
### using Mel-Frequency Cepstral Coefficients
extracted_features=[]
for index_num,row in enumerate(data):
    if index_num<115:
        featrs=features_extractor('C:\\Users\\Dyari M.shareef\\Desktop\\Audio-Auth\\cats_dogs\\'+row)
        extracted_features.append([featrs,'cat'])
    if index_num>115:
        featrs=features_extractor('C:\\Users\\Dyari M.shareef\\Desktop\\Audio-Auth\\cats_dogs\\'+row)
        extracted_features.append([featrs,'dog'])

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])

### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

### Label Encoding
###y=np.array(pd.get_dummies(y))
### Label Encoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

### No of classes
num_labels=y.shape[1]

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='audio_classification.hdf5', verbose=1, save_best_only=True)
start = datetime.now()
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)


test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])


model.predict_classes(X_test)

filename="C:\\Users\\Dyari M.shareef\\Desktop\\Audio-Auth\\cats_dogs\\cat_22.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)


mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
predicted_label=model.predict_classes(mfccs_scaled_features)
prediction_class = labelencoder.inverse_transform(predicted_label) 
prediction_class[0]





