#%% loading data

import set_workspace

from sklearn.datasets import fetch_openml

def load_data():
    mnist = fetch_openml('mnist_784',version=1)
    return mnist


mnist= load_data()
print('data loaded')

#%% helper funtions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,precision_recall_curve as prt_curve
from sklearn.base import clone, BaseEstimator
from sklearn.exceptions import NotFittedError

def showDigit(mnist_data,index):
    _piksels=mnist_data[index].reshape(28,28)
    plt.imshow(_piksels,cmap=mpl.cm.binary)

def showDigit(sigle_data):
    _piksels=sigle_data.reshape(28,28)
    plt.imshow(_piksels,cmap=mpl.cm.binary)

def splitData(mnist_dataset,train_ratio):
    train_size = np.int(train_ratio*mnist_dataset.target.size)
    return mnist_dataset.data[:train_size],mnist_dataset.target[:train_size],mnist_dataset.data[train_size:],mnist_dataset.target[train_size:] 

def get_strat_predict(model,splits,inputs,outputs):
    predicts,targets = [],[]
    sfold = StratifiedKFold(n_splits=splits,random_state=42)
    for train_index , test_index in sfold.split(inputs,outputs):
        inputs_train, inputs_test = inputs[train_index], inputs[test_index]
        outputs_train, outputs_test = outputs[train_index], outputs[test_index]
        model.fit(inputs_train,outputs_train)
        outputs_pred=model.predict(inputs_test)
        targets.append(outputs_test)
        predicts.append(outputs_pred)
    return np.asarray(predicts), np.asarray(targets)

class MyBlankEstimator(BaseEstimator):
    def fit(self,X,y=None):
        pass
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool) 

# TODO: show_roc_curve(targets,scores):

# TODO: jeszcze zoom na to
def show_prt_curve(scores,targets,treshold):
    # _prec,_rec,_tresh = prt_curve(targets,scores)
    plt.plot(_tresh,_prec[:-1])
    plt.plot(_tresh,_rec[:-1])
    _arg_prec_90 = np.argmax(_prec>treshold)
    _arg_rec_90 = np.argmax(_rec<treshold)
    _tresh_prec_90 = _tresh[_arg_prec_90]
    _tresh_rec_90 = _tresh[_arg_rec_90]
    plt.plot([-40000,_tresh_prec_90,_tresh_prec_90],[treshold,treshold,0],'r:')
    plt.plot([-40000,_tresh_rec_90,_tresh_rec_90],[treshold,treshold,0],'m:')

    plt.grid(True)
    plt.legend(['precyzja','pelnosc','precyzja_'+str(treshold),'pelnosc_'+str(treshold)],loc='best')
    plt.axis(xlim=(-40000,40000),ylim=(0,1))    
    save_fig('3_klasyfikacja_recalls')



print('funkcje przygotowane')
# %% SANDBOX
FORCE_CALC=False

# podzial danych:
if('wdata_target' not in globals() or FORCE_CALC):
    train_ratio=0.8
    train_data, train_target, test_data,test_target = splitData(mnist,train_ratio)
    wdata,wtarget,wtarget6 = train_data,train_target.astype(int),(train_target.astype(int) == 6)
     

# uczymy sie.
if('sgd_clf' not in globals() or FORCE_CALC):
    sgd_clf = SGDClassifier(max_iter=1000,tol=1e-3,random_state=42)
    sgd_clf.fit(X=wdata,y=wtarget6)
if('wtarget6_pred' not in globals() or FORCE_CALC):
    wtarget6_pred=sgd_clf.predict(wdata)
if('wtarget6_pred_strat_m' not in globals() or FORCE_CALC):
    wtarget6_pred_strat_m,wtarget6_strat_m  = get_strat_predict(model=clone(sgd_clf),splits=3,inputs=wdata,outputs=wtarget6)   #based on StratifiedFold
if('wtarget6_pred_cross' not in globals() or FORCE_CALC):
    wtarget6_pred_cross=cross_val_predict(sgd_clf,wdata,wtarget6,cv=3)
    #ciekawe: uzyskujemy gorsze wyniki, poniewaz tu pracujemy z mniejszymi zbiorami.


# selecting TODO: zrobic funkcje na to
# _rand = np.random.randint(train_target.size)
# rdigit = {
#     'data' : wdata[_rand],
#     'target' : wdata_target[_rand]}
# showDigit(rdigit['data'])
# print(sgd_clf.predict(rdigit['data'].reshape(1,-1)))

#porownujemy winiki
corrects=round(sum(wtarget6_pred==wtarget6)/len(wtarget6),2)
corrects_strat=[round(sum(x==y)/len(y),2) for x,y in zip(wtarget6_pred_strat_m,wtarget6_strat_m)]
corrects_cross=round(sum(wtarget6_pred_cross==wtarget6)/len(wtarget6),2)
print('CORRECTS[%]: ',corrects)
print('CORRECTS[%] (STRAT 3): ',corrects_strat)
print('CORRECTS[%] (CROSS): ',corrects_cross)
print('CONF MATRIX: ',confusion_matrix(wtarget6,wtarget6_pred))
print('CONF MATRIX (CROSS): ',confusion_matrix(wtarget6,wtarget6_pred_cross))
print('PRECYZJA (norm vs cross): ',round(precision_score(wtarget6_pred,wtarget6),2),round(precision_score(wtarget6_pred_cross,wtarget6)),2)
print('PELNOSC (norm vs cross): ',round(recall_score(wtarget6_pred,wtarget6),2),round(recall_score(wtarget6_pred_cross,wtarget6)),2)
print('F1 score (norm vs cross)', round(f1_score(wtarget6_pred,wtarget6),2),round(f1_score(wtarget6_pred_cross,wtarget6)),2)

#funckje decyzyjne
decisions=[round(sgd_clf.decision_function(x.reshape(1,-1))[0],2) for x in wdata[:20]]
if('wtarget6_pred_cross_score' not in globals()):
    wtarget6_pred_cross_score= cross_val_predict(sgd_clf,wdata,wtarget6,cv=3,method='decision_function')
print('DECISIONS: ',decisions)

show_prt_curve(wtarget6_pred_cross_score,wtarget6,0.8)
# show_roc_curve(targ)






# %%
# %%






# %%

# %%
