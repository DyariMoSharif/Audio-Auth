# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:12:00 2021

@author: Dyari M.shareef
"""

import pandas as pd
import librosa
import numpy as np

annotatedData=pd.read_csv("train_test_split.csv")
Xtrain=pd.concat((annotatedData['train_cat'], annotatedData['train_dog']),axis=0).dropna()
ytrain=Xtrain.str.split('_').str[0]

Xtest=pd.concat((annotatedData['test_cat'], annotatedData['test_dog']),axis=0).dropna()
ytest=Xtest.str.split('_').str[0]


def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features


def MFCCExctracting(arg):
    ### Now we iterate through every audio file and extract features 
    ### using Mel-Frequency Cepstral Coefficients
    extracted_features=[]
    for row in arg:
        featrs=features_extractor('cats_dogs\\'+row)
        extracted_features.append(featrs)
    
    #extracted_features_df=pd.DataFrame(extracted_features,columns=['feature'])
    return extracted_features

Xtrain=MFCCExctracting(Xtrain)
Xtrain=np.array(Xtrain)
Xtest=MFCCExctracting(Xtest)
Xtest=np.array(Xtest)

### Label Encoding
###y=np.array(pd.get_dummies(y))
### Label Encoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
ytrain=to_categorical(labelencoder.fit_transform(ytrain))
ytest=to_categorical(labelencoder.fit_transform(ytest))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

### No of classes
num_labels=ytrain.shape[1]

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))

###second layer
model.add(Dense(50))
model.add(Activation('relu'))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs =15
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='audio_classification.hdf5', verbose=1, save_best_only=True)
start = datetime.now()
model.fit(Xtrain, ytrain, batch_size=num_batch_size, epochs=num_epochs, validation_data=(Xtest, ytest), callbacks=[checkpointer], verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)


test_accuracy=model.evaluate(Xtest,ytest,verbose=0)
print('The accuracy: ',test_accuracy[1])


model.predict_classes(Xtest)

filename="cats_dogs\\cat_22.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
#librosa.get_duration(filename='cats_dogs\\cat_32.wav')
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)


mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
predicted_label=model.predict_classes(mfccs_scaled_features)
prediction_class = labelencoder.inverse_transform(predicted_label) 
prediction_class[0]

# for row in Xtrain:
#     print(librosa.get_duration(filename=('cats_dogs\\'+row)))