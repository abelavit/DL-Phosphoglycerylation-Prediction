# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:18:50 2023

@author: abelac
"""


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score 
import warnings
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import tensorflow as tf
import tensorflow.keras.layers as tfl
import scipy.io
from sklearn.model_selection import train_test_split

dataset_name = 'All_samples_ESM_1b_avg.mat'
Dataset_raw = scipy.io.loadmat(dataset_name)
Dataset = Dataset_raw['Final_Data']

column_names = ['Code','Feat_vec','Label','Position','Feat_mat']
DF_proteins = pd.DataFrame(columns = column_names)

## create dataframe for the dataset
for i in range(len(Dataset)):
    DF_proteins.loc[i] = Dataset[i]
DF_proteins = DF_proteins.astype({"Label": int, "Position": int}) # change column type

## separate positive and negative samples
Positive_Samples = pd.DataFrame(columns = column_names)
Negative_Samples_all = pd.DataFrame(columns = column_names)
Neg_site_of_interest = 0
Pos_site_of_interest = 0
for i in range(len(DF_proteins)):
    if DF_proteins['Label'][i] == 1:
        Positive_Samples.loc[Pos_site_of_interest] = DF_proteins.loc[i]
        Pos_site_of_interest += 1
    
    else:
        Negative_Samples_all.loc[Neg_site_of_interest] = DF_proteins.loc[i]
        Neg_site_of_interest += 1

## randomly pick negative samples to balance it with positve samples (1.5x positive samples)
Negative_Samples = Negative_Samples_all.sample(n=round(len(Positive_Samples)*1.5), random_state=42)
# saving the picked negative samples
#Negative_Samples.to_excel("BigramPGK_balanced_Negative_Samples.xlsx")

## combine positive and negative sets to make the final dataset
pos_neg_stacked = pd.concat([Positive_Samples, Negative_Samples], ignore_index=True, axis=0)

## split data into CV and independent set. 20% as independent
X = [0]*len(pos_neg_stacked)
for i in range(len(pos_neg_stacked)):
    feat = pos_neg_stacked['Feat_vec'][i]
    X[i] = feat
    #print(f'Sample {i+1} of {len(pos_neg_stacked)}')

y = pos_neg_stacked['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) # idea here is to get the index of pos_neg_stacked for independent test set and validation set. Split is stratified on y so that split has same proportions of the classes.

## create the training set using the index obtained in y_train when stratified train_test_split was done
Training_set = pd.DataFrame(columns = column_names)
for i in range(len(y_train)):
    Training_set.loc[i] = pos_neg_stacked.loc[y_train.index[i]]

## collect the features and labels of training set
X_tr = [0]*len(Training_set)
for i in range(len(Training_set)):
    feat = Training_set['Feat_vec'][i]
    X_tr[i] = feat
X_tr_arr = np.asarray(X_tr)
X_tr_reshaped = np.reshape(X_tr_arr,(X_tr_arr.shape[0],X_tr_arr.shape[2]))
y_tr = Training_set['Label'].to_numpy(dtype=float)
Y_tr_reshaped = y_tr.reshape(y_tr.size,1)

# Generate a random order of elements with np.random.permutation and simply index into the arrays Feature and label 
idx = np.random.RandomState(seed=42).permutation(len(X_tr_reshaped))
X_train,Y_train = X_tr_reshaped[idx], Y_tr_reshaped[idx]
scaler = StandardScaler()
scaler.fit(X_train) # fit on training set only
X_train = scaler.transform(X_train) # apply transform to the training set

## create the independent test set using the index obtained in y_test when stratified train_test_split was done
Independent_test_set = pd.DataFrame(columns = column_names)
for i in range(len(y_test)):
    Independent_test_set.loc[i] = pos_neg_stacked.loc[y_test.index[i]]

## collect the features and labels for independent set
X_independent = [0]*len(Independent_test_set)
for i in range(len(Independent_test_set)):
    feat = Independent_test_set['Feat_vec'][i]
    X_independent[i] = feat
X_independent_arr = np.asarray(X_independent)
X_independent_reshaped = np.reshape(X_independent_arr,(X_independent_arr.shape[0],X_independent_arr.shape[2]))
y_independent = Independent_test_set['Label'].to_numpy(dtype=float)
Y_test = y_independent.reshape(y_independent.size,1)
X_test = scaler.transform(X_independent_reshaped) # apply standardization (transform) to the test set

def CNN_Model():
    
    model = tf.keras.Sequential()
    model.add(tfl.Conv1D(128, 5, padding='same', activation='relu', input_shape=(feat_shape,1)))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.38))
    model.add(tfl.Conv1D(128, 3, padding='same',activation='relu'))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.38))
    model.add(tfl.Conv1D(64, 3, padding='same',activation='relu'))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.38))
  
    model.add(tfl.Flatten())
    
    #model.add(tfl.Dense(1000, activation='relu'))
    model.add(tfl.Dense(128, activation='relu'))
    #model.add(tfl.Dropout(0.2))
    model.add(tfl.Dense(32, activation='relu'))
    model.add(tfl.Dense(1, activation='sigmoid'))
    
    return model

feat_shape = X_train[0].size

cnn_model = CNN_Model()

learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
cnn_model.compile(optimizer=optimizer,
                   loss='binary_crossentropy',
                   metrics=['AUC'])

cnn_model.summary()

cnn_model.load_weights('ESM_1b_avg_model_weights.h5')

Inde_test_prob = cnn_model.predict(X_test)

fpr, tpr, thresholds = roc_curve(Y_test, Inde_test_prob)
inde_auc = round(roc_auc_score(Y_test, Inde_test_prob),3)

# display the metrics

print(f'Independent AUC: {inde_auc}')

# plot ROC curve
legend = 'AUC = ' + str(inde_auc)
pyplot.figure(figsize=(12,8))
pyplot.plot([0,1], [0,1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.', label=legend)
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()
















'''
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score 
import warnings
import pickle
import copy
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import tensorflow as tf
import tensorflow.keras.layers as tfl
import scipy.io
from sklearn.model_selection import train_test_split

dataset_name = 'All_samples_ESM_1b_avg.mat'
Dataset_raw = scipy.io.loadmat(dataset_name)
Dataset = Dataset_raw['Final_Data']

column_names = ['Code','Feat_vec','Label','Position','Feat_mat']
DF_proteins = pd.DataFrame(columns = column_names)

## create dataframe for the dataset
for i in range(len(Dataset)):
    DF_proteins.loc[i] = Dataset[i]
DF_proteins = DF_proteins.astype({"Label": int, "Position": int}) # change column type

## separate positive and negative samples
Positive_Samples = pd.DataFrame(columns = column_names)
Negative_Samples_all = pd.DataFrame(columns = column_names)
Neg_site_of_interest = 0
Pos_site_of_interest = 0
for i in range(len(DF_proteins)):
    if DF_proteins['Label'][i] == 1:
        Positive_Samples.loc[Pos_site_of_interest] = DF_proteins.loc[i]
        Pos_site_of_interest += 1
    
    else:
        Negative_Samples_all.loc[Neg_site_of_interest] = DF_proteins.loc[i]
        Neg_site_of_interest += 1

## randomly pick negative samples to balance it with positve samples (1.5x positive samples)
Negative_Samples = Negative_Samples_all.sample(n=round(len(Positive_Samples)*1.5), random_state=42)
# saving the picked negative samples
#Negative_Samples.to_excel("BigramPGK_balanced_Negative_Samples.xlsx")

## combine positive and negative sets to make the final dataset
pos_neg_stacked = pd.concat([Positive_Samples, Negative_Samples], ignore_index=True, axis=0)

## split data into CV and independent set. 20% as independent
X = [0]*len(pos_neg_stacked)
for i in range(len(pos_neg_stacked)):
    feat = pos_neg_stacked['Feat_vec'][i]
    X[i] = feat
    #print(f'Sample {i+1} of {len(pos_neg_stacked)}')

y = pos_neg_stacked['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) # idea here is to get the index of pos_neg_stacked for independent test set and validation set. Split is stratified on y so that split has same proportions of the classes.
# saving the CV and independent set indices
'''
df1 = pd.DataFrame(y_train)
df1.to_excel("BigramPGK_ytrain.xlsx")
df2 = pd.DataFrame(y_test)
df2.to_excel("BigramPGK_ytest.xlsx")
'''

## create the training set using the index obtained in y_train when stratified train_test_split was done
Training_set = pd.DataFrame(columns = column_names)
for i in range(len(y_train)):
    Training_set.loc[i] = pos_neg_stacked.loc[y_train.index[i]]

## collect the features and labels of training set
X_tr = [0]*len(Training_set)
for i in range(len(Training_set)):
    feat = Training_set['Feat_vec'][i]
    X_tr[i] = feat
X_tr_arr = np.asarray(X_tr)
X_tr_reshaped = np.reshape(X_tr_arr,(X_tr_arr.shape[0],X_tr_arr.shape[2]))
y_tr = Training_set['Label'].to_numpy(dtype=float)
Y_tr_reshaped = y_tr.reshape(y_tr.size,1)

# Generate a random order of elements with np.random.permutation and simply index into the arrays Feature and label 
idx = np.random.RandomState(seed=42).permutation(len(X_tr_reshaped))
X_train,Y_train = X_tr_reshaped[idx], Y_tr_reshaped[idx]
scaler = StandardScaler()
scaler.fit(X_train) # fit on training set only
X_train = scaler.transform(X_train) # apply transform to the training set

## create the independent test set using the index obtained in y_test when stratified train_test_split was done
Independent_test_set = pd.DataFrame(columns = column_names)
for i in range(len(y_test)):
    Independent_test_set.loc[i] = pos_neg_stacked.loc[y_test.index[i]]

## collect the features and labels for independent set
X_independent = [0]*len(Independent_test_set)
for i in range(len(Independent_test_set)):
    feat = Independent_test_set['Feat_vec'][i]
    X_independent[i] = feat
X_independent_arr = np.asarray(X_independent)
X_independent_reshaped = np.reshape(X_independent_arr,(X_independent_arr.shape[0],X_independent_arr.shape[2]))
y_independent = Independent_test_set['Label'].to_numpy(dtype=float)
Y_test = y_independent.reshape(y_independent.size,1)
X_test = scaler.transform(X_independent_reshaped) # apply standardization (transform) to the test set

def CNN_Model():
    
    model = tf.keras.Sequential()
    model.add(tfl.Conv1D(128, 5, padding='same', activation='relu', input_shape=(feat_shape,1)))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.38))
    model.add(tfl.Conv1D(128, 3, padding='same',activation='relu'))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.38))
    model.add(tfl.Conv1D(64, 3, padding='same',activation='relu'))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.38))
  
    model.add(tfl.Flatten())
    
    #model.add(tfl.Dense(1000, activation='relu'))
    model.add(tfl.Dense(128, activation='relu'))
    #model.add(tfl.Dropout(0.2))
    model.add(tfl.Dense(32, activation='relu'))
    model.add(tfl.Dense(1, activation='sigmoid'))
    
    return model

feat_shape = X_train[0].size

cnn_model = CNN_Model()

learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
cnn_model.compile(optimizer=optimizer,
                   loss='binary_crossentropy',
                   metrics=['AUC'])

cnn_model.summary()

cnn_model.load_weights('ESM_1b_avg_model_weights.h5')

eval_result = cnn_model.evaluate(X_test, Y_test)

print(f"test loss: {round(eval_result[0],4)}, test auc: {round(eval_result[1],4)}")
Inde_test_prob = cnn_model.predict(X_test)

def round_based_on_thres(probs_to_round, set_thres):
    for i in range(len(probs_to_round)):
        if probs_to_round[i] <= set_thres:
            probs_to_round[i] = 0
        else:
            probs_to_round[i] = 1
    return probs_to_round

# calculate the metrics
set_thres = 0.5
copy_Probs_inde = copy.copy(Inde_test_prob)
round_based_on_thres(copy_Probs_inde, set_thres)
fpr, tpr, thresholds = roc_curve(Y_test, Inde_test_prob)
inde_auc = round(roc_auc_score(Y_test, Inde_test_prob),4)
inde_pre = round(precision_score(Y_test, copy_Probs_inde),4)
cm = confusion_matrix(Y_test, copy_Probs_inde) # for acc, sen, and spe calculation
total_preds=sum(sum(cm))
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]
inde_sen = round(TP/(TP+FN),4)
inde_spe = round(TN/(TN+FP),4)
acc_independent = round(((TN+TP)/(total_preds)),4)

# display the metrics
print(f'Independent Sen: {inde_sen}')
print(f'Independent Spe: {inde_spe}')
print(f'Independent Pre: {inde_pre}')
print(f'Independent Acc: {acc_independent}')
print(f'Independent AUC: {inde_auc}')

# plot ROC curve
legend = 'AUC = ' + str(inde_auc)
pyplot.figure(figsize=(12,8))
pyplot.plot([0,1], [0,1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.', label=legend)
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

pickle.dump(Inde_test_prob,open("y_independent_ESM_1b_1DCNN.dat","wb"))
'''
