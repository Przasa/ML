#%% loading data
from sklearn.datasets import fetch_openml

def load_data():
    mnist = fetch_openml('mnist_784',version=1)
    return mnist

mnist= load_data()
print('data loaded')



#%% helper funtions
import set_workspace as helpers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,precision_recall_curve as prt_curve
from sklearn.base import clone, BaseEstimator
from sklearn.exceptions import NotFittedError

# selecting TODO: zrobic funkcje na to
# _rand = np.random.randint(train_target.size)
# rdigit = {
#     'data' : wdata[_rand],
#     'target' : wdata_target[_rand]}
# showDigit(rdigit['data'])
# print(sgd_clf.predict(rdigit['data'].reshape(1,-1)))

def showDigit(mnist_data,index):
    _piksels=mnist_data[index].reshape(28,28)
    plt.imshow(_piksels,cmap=mpl.cm.binary)

def showDigit(sigle_data):
    _piksels=sigle_data.reshape(28,28)
    plt.imshow(_piksels,cmap=mpl.cm.binary)

def splitData(mnist_dataset,train_ratio):   # TODO: juz nie potrzebne
    train_size = np.int(train_ratio*mnist_dataset.target.size)
    return mnist_dataset.data[:train_size],mnist_dataset.target[:train_size],mnist_dataset.data[train_size:],mnist_dataset.target[train_size:] 


def get_strat_predict(model,splits,inputs,outputs):
    predicts,targets = [],[]
    sfold = StratifiedKFold(n_splits=splits,random_state=42,shuffle=True)
    for train_index , test_index in sfold.split(inputs,outputs):
        inputs_train, inputs_test = inputs[train_index], inputs[test_index]
        outputs_train, outputs_test = outputs[train_index], outputs[test_index]
        model.fit(inputs_train,outputs_train)
        outputs_pred=model.predict(inputs_test)
        targets.append(outputs_test)
        predicts.append(outputs_pred)
    return np.asarray(predicts), np.asarray(targets)



# TODO: jeszcze zoom na to
def show_prt_curve(scores,targets,treshold):
    _prec,_rec,_tresh = prt_curve(targets,scores)
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
    helpers.save_fig('3_klasyfikacja_recalls')


class MyBlankEstimator(BaseEstimator):
    def fit(self,X,y=None):
        pass
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool) 

class StoreData():
    def __init__(self,entire_dataset,train_ratio):
        train_size = np.int(train_ratio*entire_dataset.target.size)
        self.train_data, self.train_target, self.test_data, self.test_target=entire_dataset.data[:train_size],entire_dataset.target[:train_size],entire_dataset.data[train_size:],entire_dataset.target[train_size:] 
    # def __init__(self,train_data,train_target,test_data,test_target):
    #     self.train_data, self.train_target, self.test_data, self.test_target = train_data,train_target,test_data,test_target
    def getTrainData(self):
        return self.train_data, self.train_target
    def getTestData(self):
        return self.test_data, self.test_target
    def getData(self):
        return self.train_data, self.train_target,self.test_data, self.test_target




print('funkcje przygotowane')
# %% SANDBOX
FORCE_CALC=False

# podzial danych:
if('train_data' not in globals() or FORCE_CALC):
    train_ratio=0.8
    data_stored = StoreData(entire_dataset=mnist,train_ratio=0.8)
    train_data, train_target, test_data, test_target = data_stored.getData()
    train_target6,test_target6  = train_target.astype(int)==6,test_target.astype(int)==6

# uczymy sie.
if('sgd_clf' not in globals() or FORCE_CALC):
    sgd_clf = SGDClassifier(max_iter=1000,tol=1e-3,random_state=42)
    sgd_clf.fit(X=train_data,y=train_target6)
if('test_pred6' not in globals() or FORCE_CALC):
    test_pred6=sgd_clf.predict(test_data)
if('test_pred6_strats' not in globals() or FORCE_CALC):
    test_pred6_strats,test_target6_strats  = get_strat_predict(model=clone(sgd_clf),splits=3,inputs=test_data,outputs=test_target6)   #based on StratifiedFold
if('test_pred6_cross' not in globals() or FORCE_CALC):
    test_pred6_cross=cross_val_predict(sgd_clf,test_data,test_target6,cv=3)
if('test_pred6_rfor' not in globals() or FORCE_CALC):
    rfor_clf = RandomForestClassifier(n_estimators=100,random_state=42)
    rfor_clf.fit(train_data,train_target)
    test_pred6_rfor= rfor_clf.predict(test_data)


    #ciekawe: uzyskujemy gorsze wyniki, poniewaz tu pracujemy z mniejszymi zbiorami.


# TODO: moze jakas funkcja/klasa raportujaca? moze w set_workspace
#porownujemy winiki
corrects=round(sum(test_pred6==test_target6)/len(test_target6),4)
corrects_strat=[round(sum(x==y)/len(y),4) for x,y in zip(test_pred6_strats,test_target6_strats)]
corrects_cross=round(sum(test_pred6_cross==test_target6)/len(test_target6),4)
corrects_rfor= round(sum(test_target==test_pred6_rfor)/len(test_target),4)

print('CORRECTS[%] (norm, strat, cross,rfor): ',corrects, max(corrects_strat), corrects_cross,corrects_rfor)
print('CONF MATRIX: (norm, cross)') ; print(confusion_matrix(test_target6,test_pred6));print(confusion_matrix(test_target6,test_pred6_cross))
# TODO jeszcze todac stratsy do conf_matrix
print('PRECYZJA (norm vs cross): ',precision_score(test_pred6,test_target6),precision_score(test_pred6_cross,test_target6))
print('PELNOSC (norm vs cross): ',recall_score(test_pred6,test_target6),2,recall_score(test_pred6_cross,test_target6))
print('F1 score (norm vs cross)', f1_score(test_pred6,test_target6),f1_score(test_pred6_cross,test_target6))

#funckje i krzywe decyzyjne
# TODO: Warto dodac jeszcze cos 
if('wtarget6_pred_cross_score' not in globals() or True):
    test_target6_pred_score=[round(sgd_clf.decision_function(x.reshape(1,-1))[0],2) for x in test_data[:20]]
    test_target6_pred_cross_score= cross_val_predict(sgd_clf,test_data,test_target6,cv=3,method='decision_function')
print('DECISIONS (norm vs cross): ',test_target6_pred_score[:10],test_target6_pred_cross_score[0:10])

# TODO: dodac porownanie krzywych z roznych modeli
show_prt_curve(test_target6_pred_cross_score,test_target6,0.8)
# TODO: show_roc_curve(targets,scores):




# %%


from sklearn.ensemble import RandomForestClassifier

if('test_pred6_rfor' not in globals() or FORCE_CALC):
    rfor_clf = RandomForestClassifier(n_estimators=100,random_state=42)
    rfor_clf.fit(train_data,train_target6)
    test_pred6_rfor= rfor_clf.predict(test_data)

corrects_rfor= round(sum(test_target==test_pred6_rfor)/len(test_target),4)
confusion_matrix(test_target6,test_pred6_rfor)
precision_score(test_pred6_rfor,test_target6)
recall_score(test_pred6_rfor,test_target6)
f1_score(test_pred6_rfor,test_target6)
# print(scores_rfor)





# %%

# %%
