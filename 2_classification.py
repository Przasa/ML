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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,precision_recall_curve as prt_curve
from sklearn.base import clone, BaseEstimator
from sklearn.exceptions import NotFittedError

# selecting TODO: zrobic funkcje na to
def pickRandom(datas,targets):
    _rand = np.random.randint(targets.size)
    return datas[_rand],targets[_rand]
    # rdigit = {
    #     'data' : wdata[_rand],
    #     'target' : wdata_target[_rand]}
    # showDigit(rdigit['data'])
    # print(sgd_clf.predict(rdigit['data'].reshape(1,-1)))

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
    rfor_clf.fit(train_data,train_target6)
    test_pred6_rfor= rfor_clf.predict(test_data)


    #ciekawe: uzyskujemy gorsze wyniki, poniewaz tu pracujemy z mniejszymi zbiorami.


# TODO: moze jakas funkcja/klasa raportujaca? moze w set_workspace
#porownujemy winiki
corrects=round(sum(test_pred6==test_target6)/len(test_target6),4)
corrects_strat=[round(sum(x==y)/len(y),4) for x,y in zip(test_pred6_strats,test_target6_strats)]
corrects_cross=round(sum(test_pred6_cross==test_target6)/len(test_target6),4)
corrects_rfor= round(sum(test_target6==test_pred6_rfor)/len(test_target6),4)
cm_norm=confusion_matrix(test_target6,test_pred6).reshape(1,4)
cm_cross=confusion_matrix(test_target6,test_pred6_cross).reshape(1,4)
cm_strat=confusion_matrix(test_target6_strats[np.argmax(corrects_strat)],test_pred6_strats[np.argmax(corrects_strat)]).reshape(1,4)
cm_rfor=confusion_matrix(test_target6,test_pred6_rfor).reshape(1,4)
ps_norm=round(precision_score(test_target6,test_pred6),4)
ps_cross=round(precision_score(test_target6,test_pred6_cross),4)
ps_strat=round(precision_score(test_target6_strats[np.argmax(corrects_strat)],test_pred6_strats[np.argmax(corrects_strat)]),4)
ps_rfor=round(precision_score(test_target6,test_pred6_rfor),4)
rs_norm=round(recall_score(test_target6,test_pred6),4)
rs_cross=round(recall_score(test_target6,test_pred6_cross),4)
rs_strat=round(recall_score(test_target6_strats[np.argmax(corrects_strat)],test_pred6_strats[np.argmax(corrects_strat)]),4)
rs_rfor=round(recall_score(test_target6,test_pred6_rfor),4)
f1_norm=round(f1_score(test_target6,test_pred6),4)
f1_cross=round(f1_score(test_target6,test_pred6_cross),4)
f1_strat=round(f1_score(test_target6_strats[np.argmax(corrects_strat)],test_pred6_strats[np.argmax(corrects_strat)]),4)
f1_rfor=round(f1_score(test_target6,test_pred6_rfor),4)

print('CORRECTS[%]:','\n\tnorm:\t',corrects,'\n\tcross:\t', corrects_cross,'\n\tstrat:\t', max(corrects_strat),'\n\trfor:\t',corrects_rfor)
print('CONFUSION MATRIX [TN(=wylapane zle), FP(=brudy), FN(=stracone), TP(=zlapane dobre)]:' , '\n\tnorm:\t',cm_norm,'\n\tcross:\t',cm_cross,'\n\tstrat:\t',cm_strat,'\n\trfor:\t',cm_rfor)
print('PRECISION: tp / (tp + fp)','\n\tnorm:\t',ps_norm,'\n\tcross:\t',ps_cross,'\n\tstart:\t',ps_strat,'\n\trfor:\t',ps_rfor)
print('RECALL: tp / (tp + fn)','\n\tnorm:\t',rs_norm,'\n\tcross:\t',rs_cross,'\n\tstart:\t',rs_strat,'\n\trfor:\t',rs_rfor)
print('F1 SCORE (F1 = 2 * (precision * recall) / (precision + recall) ):','\n\tnorm:\t',f1_norm,'\n\tcross:\t',f1_cross,'\n\tstart:\t',f1_strat,'\n\trfor:\t',f1_rfor)

#funckje i krzywe decyzyjne
# TODO: Warto dodac jeszcze cos 
if('wtarget6_pred_cross_score' not in globals() or True):
    test_target6_pred_score=[round(sgd_clf.decision_function(x.reshape(1,-1))[0],2) for x in test_data[:20]]
    test_target6_pred_cross_score= cross_val_predict(sgd_clf,test_data,test_target6,cv=3,method='decision_function')
    # np.array(test_target6_pred_score) 
print('DECISIONS (norm vs cross): ',test_target6_pred_score[:10],test_target6_pred_cross_score[0:10])
# TODO: dodac porownanie krzywych z roznych modeli
show_prt_curve(test_target6_pred_cross_score,test_target6,0.8)
# TODO: show_roc_curve(targets,scores):






# %% Klasyfikacja wieloklasowa
# inene metody tez mozesz tak potraktowac.

# 0. SVC clf
# 0. One vs Rest
# 1. sross_val_score(sgd_clf,xscaled,...)
# 2. cross_val_predict(sgd_clf,xscaled,...)
# 3. confusion_matrix(^)
# 4. plot conf_matrix (zwykly i przeskalowany)


#TODO: tez model tez zestawic z innymi (jak bylo)
from sklearn.svm import SVC 
from sklearn.multiclass import OneVsRestClassifier  # takie rzeczy to tylko dla multiclass
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

_rx,_ry=pickRandom(test_data,test_target)
# showDigit(_rx)

# # TODO: 1000 sampli to za malo na dobre uczenie. Przygotuj sie na troche wiecej.
if('svc_mclf' not in globals() or FORCE_CALC):
    svc_mclf = SVC(gamma='auto',random_state=42)
    svc_mclf.fit(train_data[:1000],train_target[:1000])

    ry_pred_svc=svc_mclf.predict([_rx])
    scores_svc=svc_mclf.decision_function([_rx])    
    ry_pred_svc_form=svc_mclf.classes_[np.argmax(scores_svc)]

# TODO: moze opakowac jakas metode do tych rzeczy
# #opakowanie od OVR (metoda OVR jest ustawiana automatycznie, dla klasyfiacji wielokasowej, ale mozno to wywolac jawnie.)
if('svc_mclf_ovr' not in globals() or FORCE_CALC):
    svc_mclf_ovr = OneVsRestClassifier(SVC())
    svc_mclf_ovr.fit(train_data[:1000],train_target[:1000]) 

    ry_pred_svc_ovr= svc_mclf_ovr.predict([_rx])
    scores_svc_ovr=svc_mclf_ovr.decision_function([_rx]) 
    classes_svc_ovr = svc_mclf_ovr.estimators_ 

if('sgd_mclf' not in globals() or FORCE_CALC):
    sgd_mclf = SGDClassifier(random_state=42)
    sgd_mclf.fit(X=train_data[:1000],y=train_target[:1000])
    
    ry_pred_sgd=sgd_mclf.predict([_rx])
    scores_sgd = sgd_mclf.decision_function([_rx])
    ry_pred_sgd_form=sgd_mclf.classes_[np.argmax(scores_sgd)]
    


print("PREDICT COMPARISION:\n\tSVC:\t\t",ry_pred_svc[0],"\n\tOVR(SVC):\t",ry_pred_svc_ovr[0],"\n\tSGD:\t",ry_pred_sgd[0])
print("SCORES COMPARISION:\n\tSVC:\t",scores_svc,"\n\tOVR(SVC):\t",scores_svc_ovr,"\n\tOVR(SVC):\t",scores_sgd)

#::::::::::::::analiza bledo:::::::::::::::

#skalowanie potrzeben do pracy z cross_val_score i cross_val_predict
# ponadto, skalowanie pozwala na uzyskanei lepdzych wynikow 
# TODO: porownac z uczeniem bez skalowania
scaler = StandardScaler()
train_data_scaled=scaler.fit_transform(train_data.astype(np.float64))
test_data_scaled = scaler.fit_transform(test_data.astype(np.float64))

# sprawdzinay krzyzowe, przewiduje wyniki dla partii, wobec ktorych nie bylo jeszcze uczenia (odklada na pozniej)
# ...ale nie widze sposobu zeby uzyc tego sprawdziany dla danych testowych (moze dac wyuczony klasyfikator? ... nie raczej nie ma to znaczenie)
if('train_score_crosssvc' not in globals() or FORCE_CALC):
    train_score_crosssvc= cross_val_score(svc_mclf,X=train_data_scaled[:2000],y=train_target[:2000],cv=4,scoring="accuracy")
    train_pred_crosssvc= cross_val_predict(svc_mclf,X=train_data_scaled[:2000],y=train_target[:2000],cv=4)
    # trenowanie od zera daje troszeczke gorsze wyniki (~0.11%)
    # score_cross_svc2= cross_val_score(SVC(),X=train_data_scaled[:2000],y=train_target[:2000],cv=4,scoring="accuracy")
    # pred_cross_svc2= cross_val_predict(SVC(),X=train_data_scaled[:2000],y=train_target[:2000],cv=4)

#1.04.2021: Teraz tu jestes. dzialaj odtad.

conf_mx = confusion_matrix(train_target[:2000],train_pred_crosssvc)
conf_mx_norm=conf_mx/conf_mx.sum(axis=1,keepdims=True)      #normalizacja ze wzgledu na liczb probek
np.fill_diagonal(conf_mx_norm,0)                            
plt.matshow(conf_mx_norm,cmap=plt.cm.gray) #bez CMAP, rysunek jest kolorowy


print("OK")


# %% Klasyfikacja wieloetykietowa (jeden sampel ma wiele etykiet)
from sklearn.neighbors import KNeighborsClassifier

train_target_odd  = (train_target.astype(int) % 2) ==1
train_target_big = (train_target.astype(int) >= 7)
train_target_mlabel = np.c_[train_target_odd,train_target_big]   # moze opakowac w funkcje

if('kn_clf' not in globals() or FORCE_CALC):
    kn_clf = KNeighborsClassifier()
    kn_clf.fit(train_data[:1000],train_target_mlabel[:1000])
    ry_pred_kn=kn_clf.predict([_rx])

if('kn_clf_cross' not in globals() or FORCE_CALC):
    train_target_mlabel_pred_kn= cross_val_predict(kn_clf,X=train_data[:1000],y=train_target_mlabel[:1000])
    f1_mlabel_kn=f1_score(train_target_mlabel[:1000],train_target_mlabel_pred_kn[:1000],average="weighted") # hipeparametr "avarage" do ustalenia sily wag


def makeSomeNoise(input):#   :)
    if(len(input.shape)>1):
        noise = np.random.randint(0,100,(len(input),len(input[0])))
    else:        
        noise = np.random.randint(0,100,len(input))

    return input+noise


# klasyfikacja wielowyjciowo-wieloklasowa
if('pred_kn' not in globals() or FORCE_CALC):
    a,b = pickRandom(test_data,test_target)
    _X = makeSomeNoise(train_data)
    _y = train_data
    _sX=makeSomeNoise(a)
    _sy=a
    
    kn_clf_mo = KNeighborsClassifier()
    kn_clf_mo.fit(_X[:1000],_y[:1000])
    pred_kn= kn_clf_mo.predict([_sX])

plt.subplot(121); showDigit(_sX)
plt.subplot(122); showDigit(pred_kn)




print('Zrobione')
# %%
