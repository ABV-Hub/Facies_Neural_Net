#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:07:16 2019

@author: reinaldosabbagh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint 
from sklearn.model_selection import train_test_split

from utils import (make_facies_log_plot,
                  augment_features,
                  facies_colors)

wells=pd.read_csv('/Users/reinaldosabbagh/Documents/facies_classification/training_data.csv') # Wells to train 

wells_blind=pd.read_csv('/Users/reinaldosabbagh/Documents/facies_classification/wells_blind.csv') # Wells for validation 

wells=wells[wells['Well Name']!='SHANKLE'] # Shankle is validation well

feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
well = wells['Well Name'].values
depth = wells['Depth'].values
X=wells[feature_names].values
wells['Facies']=wells['Facies'].values-1
correct_facies=wells['Facies']


X_val=wells_blind[feature_names].values
depth_val= wells_blind['Depth'].values
well_val=wells_blind['Well Name']
wells_blind['Facies']=wells_blind['Facies'].values-1
correct_facies_val=wells_blind['Facies']

#Feature Augmentation
aug, padded_rows = augment_features(X, well, depth)
aug_val, padded_rows_val= augment_features(X_val, well_val, depth_val)

#Feature Scaling 
scaler = StandardScaler()  
scaler.fit(aug)  
scaled_features = scaler.transform(aug)  
scaled_features_val = scaler.transform(aug_val)  
#Train Test Split
X_train, X_test, y_train, y_test = train_test_split( scaled_features, correct_facies, test_size=0.3, random_state=45)

#Create the Neural Network Model
model = tf.keras.Sequential([
    keras.layers.Dense(45,input_dim=(28), activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(55, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(9,activation=tf.nn.sigmoid)])
    
epochs=650
decay_rate = 0.005 / epochs
momentum = 0.9
sgd = tf.keras.optimizers.SGD(lr=0.005, decay= decay_rate, momentum=momentum, nesterov=True)
 
model.compile(loss='sparse_categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

filepath="/Users/reinaldosabbagh/Documents/facies_classification/best.h5"
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=False)

# Fir the model 

history=model.fit(X_train,y_train,epochs=epochs,batch_size=256,verbose=0,callbacks=[checkpointer],validation_data=(X_test,y_test))

# summarize history for loss

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#Load rhe best model 
from keras.models import load_model
from keras.initializers import glorot_uniform
from keras.utils import CustomObjectScope
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    best_model=load_model(filepath)


print(best_model.evaluate(scaled_features_val,correct_facies_val))


facies_predicted=best_model.predict(scaled_features_val)

test=np.argmax(facies_predicted, axis=1)
wells_blind['Facies_pred']=test


make_facies_log_plot(
    wells_blind[wells_blind['Well Name'] == 'SHANKLE'],
    facies_colors,logs_pred=True)

from sklearn.metrics import confusion_matrix
confusion_matrix(test, correct_facies_val)
from sklearn.metrics import classification_report 
print(classification_report(test, correct_facies_val) )





