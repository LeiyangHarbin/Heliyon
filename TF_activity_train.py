# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:12:13 2022

@author: Administor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from pandas import read_csv
from sklearn.metrics import precision_recall_curve, auc
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection  import GridSearchCV
from sklearn.preprocessing import StandardScaler

seed=7
np.random.seed(seed)

OV_train = pd.read_csv('OV_train.tsv',header='infer',sep="\t")
train_label = pd.read_csv('label_train.tsv',header='infer',sep="\t")

#TCGA_dat = TCGA_TF_activities[TF]
train_dat =StandardScaler().fit_transform(OV_train)
#X = TCGA_dat
#Y = TCGA_label

#################################
X = train_dat
X = np.array(X)
Y = train_label
Y = np.array(Y)



input_size =1310

def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(1000,input_dim=input_size,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(250,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model)
optimizers = ['adam','SGD']

epochs_1 = list([50,100,150])
batches = list([5,10,20])
param_grid = dict(optimizer=optimizers,epochs=epochs_1,batch_size=batches)

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
canshu = grid_result.best_params_
opt = canshu.get('optimizer')
epo = canshu.get('epochs')
bat = canshu.get('batch_size')

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

model = KerasClassifier(build_fn=create_model, nb_epoch=epo, batch_size=bat)


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print('Accuracy: %.2f (%.2f) MSE' % (results.mean(), results.std()))
print(results.mean())
result1 = pd.DataFrame(results)
result_mean = results.mean()
result1.iloc[9,0] = result_mean
result1.index = [1,2,3,4,5,6,7,8,9,'mean']
result1.columns = ['cross_val_score']
result1.to_csv('cross_val_score.tsv',sep="\t")



##########################
X = train_dat
X = np.array(X)
Y = train_label
Y = np.array(Y)

Y[Y == 1] = 0
Y[Y == 2] = 1

axes = plt.subplots(1,1,figsize=(5, 5))

predictor = KerasClassifier(build_fn=create_model, nb_epoch=epo, batch_size=bat)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

y_real = []
y_proba = []
for i, (train_index, test_index) in enumerate(kfold.split(X,Y)):
    Xtrain, Xtest = X[train_index], X[test_index]
    Ytrain, Ytest = Y[train_index], Y[test_index]
    predictor.fit(Xtrain, Ytrain)
    pred_proba = predictor.predict_proba(Xtest)
    #precision, recall, _ = precision_recall_curve(Ytest, pred_proba[:,1])
    #lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
    #axes[1].step(recall, precision, label=lab)
    y_real.append(Ytest)
    y_proba.append(pred_proba[:,1])

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
lab = 'Overall AUC=%.4f' % (auc(recall, precision))


axes[1].step((1-precision),recall,label=lab, lw=2, color='red')
axes[1].set_xlabel('1-Precision')
axes[1].set_ylabel('Recall')
axes[1].legend(loc='lower right', fontsize='small')

plt.show()

axes[0].savefig('result.pdf')

##########################################
# make predictions
from sklearn.preprocessing import StandardScaler
estimator = KerasClassifier(build_fn=create_model, epochs=epo, batch_size=bat)
estimator.fit(X, Y)


test_dat=pd.read_csv('OV_test.tsv',header='infer',sep="\t")



#####################################
# make predictions
test_scale= StandardScaler().fit_transform(test_dat)
pred = estimator.predict(test_dat)
#pred = estimator.predict(METABRIC_sig_feature)
pred_test = pd.DataFrame(pred)
pred_test.to_csv('test_predict_lable.tsv',sep=",",header=True)
################

