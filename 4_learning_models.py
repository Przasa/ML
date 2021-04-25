# %% initials

# Xs=     (x0=1,  x1=X)
#         1     0
#         1     0.1
#         1     0.2
#         1     0.3
#         1     ...
#         1     2
# theta_cform =  a0=15  
#                a1=2.8 
# y(x)  Xs*theta_cform=a0*x0+a1*x1 =a0 + a1*X
# y(0)          =a0*1+a1*0
# y(1.1)        =a0*1+a1*1.1

FORCE_CALC=False

import set_workspace as wspace
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.linalg

#%% funckje
def createRandomLinear(length):
    X = 2 * np.random.rand(length, 1)        # nie myl X z calym zestawem cech X_vec. X to tylko jedna cecha (tj X0=1, X1=X) 
    y = 15 + 3 * X + np.random.randn(length, 1) #randn to szum gausowski
    return X,y

def calc_theta_corm(X,y):
    X_vec=np.c_[np.ones((len(X),1)),X]
    theta_cform=np.linalg.inv(X_vec.T.dot(X_vec)).dot(X_vec.T).dot(y)
    return theta_cform

def calc_theta_linreg(X,y):
    X_vec=np.c_[np.ones((len(X),1)),X]
    lin_reg=LinearRegression()
    lin_reg.fit(X=X_vec,y=y)
    theta_linreg=np.r_[[lin_reg.intercept_,lin_reg.coef_[0][1:]]]
    # yp_linreg=lin_reg.predict(X_vec)
    return theta_linreg #,yp_linreg

def calc_theta_svd(X,y):
    X_vec=np.c_[np.ones((len(X),1)),X]
    theta_svd ,_esiduals, _rank, _s =np.linalg.lstsq(X_vec, y, rcond=1e-6)
    return theta_svd

def predict_by_theta(Xs,theta):
    Xs_vec=np.c_[np.ones((len(Xs),1)),Xs]
    return Xs_vec.dot(theta)

class Samples():
    def __init__(self,X,y,name,plot_conf):
        self.X,self.y,self.name,self.plot_conf=X,y,name,plot_conf
    
class Ploter():

    lines_list = []
    samples_list=[]
    nfig=1    
    def __init__(self):
        self.lines_list=[]
        self.samples_list=[]
    def addSamples(self,X,y,name,plot_conf):
        samples_obj = Samples(X,y,name,plot_conf)
        self.samples_list.append(samples_obj)
    def makePlot(self,zoomed=False):
        plt.clf()
        plt.figure(self.nfig)
        for sample in self.samples_list:
            plt.plot(sample.X,sample.y,sample.plot_conf,label=sample.name)
        plt.grid(True)
        plt.legend(loc='upper left')

        if(zoomed):
            plt.title(label=str(self.nfig)+': prediction (zoomed)')
            plt.axis([1,1.1,17,19])
        else:
            plt.title(label=str(self.nfig)+': prediction')

        self.nfig=self.nfig+1
        plt.show()
    

class AnalyzePrinter():
    
    def __init__(self):
        self.theta_dictlist=[]
        pass
    def addTheta(self,theta):
        self.theta_dictlist.append(theta)
    def compare_thetas(self):
        print("Theta comparision:")
        for theta in self.theta_dictlist:
            print("\t"+theta['name']+":\n"+str(theta['data']))


def makeAnalyze():
    ploter = Ploter()
    ploter.addSamples(X,y,name='samples',plot_conf="b.")
    ploter.addSamples(Xs,yp_cform,name='prediction (closed form)',plot_conf="r-")
    ploter.addSamples(Xs,yp_linreg,name='prediction (linear reg)',plot_conf="g-")
    ploter.addSamples(Xs,yp_svd,name='prediction (svd)',plot_conf="b-")
    ploter.makePlot()
    ploter.makePlot(zoomed=True)

    aprinter = AnalyzePrinter()
    for x in [{'name':'theta_cform','data':theta_cform},{'name':'theta_linreg','data':theta_linreg},{'name':'theta_svd','data':theta_svd}]: 
        aprinter.addTheta(x)
    aprinter.compare_thetas()

print("code prepared")

def batchGradientDescent(X,y):
#wsadowy gradient prosty 
# -> trzeba od razu wszystkie dane, ale niezly dla wielu cech (liczy je jednoczesnie)
#jednak wartosci oscyluja. ale musisz uwazasz zeby uklad byl wciaz stabilny :)
#to oznacza ze wektor gradientow przyjmuje tez wartosci ujemne (gdy przestrzelamy)
    eta=0.007
    n_iterations=1000
    m= len(X)               #100
    # m=samples_qty         #100


    theta = np.random.randn(2,1)
    X_vec=np.c_[np.ones((len(X),1)),X]
    glist=[]
    tlist=[]
    for iterations in range(n_iterations):
        gradients = 2/m + X_vec.T.dot(X_vec.dot(theta)-y)
        theta = theta - eta*gradients
        glist.append(gradients.T[0])
        tlist.append(theta.T[0])
        tarr=np.asarray(tlist)
        garr=np.asarray(glist)
        if np.abs(gradients[0])< 0.001 and np.abs(gradients[1]) < 0.001:
            break

    plt.figure(1)
    plt.plot(tarr[:,0],tarr[:,1],'b')
    plt.grid(True)

    plt.figure(2)
    plt.plot(garr[:,0],garr[:,1],'b')
    plt.grid(True)

def closedForms(X,y):
    theta_cform, theta_linreg, theta_svd = calc_theta_corm(X,y),calc_theta_linreg(X,y),calc_theta_svd(X,y)
    yp_cform,yp_linreg,yp_svd = predict_by_theta(Xs,theta_cform),predict_by_theta(Xs,theta_linreg),predict_by_theta(Xs,theta_svd)
    
    makeAnalyze()       #troche przerost formy z ta funckja


# %%
samples_qty=100
#TODO zrob cos takiego dla wielu cech, nie tylko jednej z zerową
if('X' not in globals() or 'y' not in globals() or FORCE_CALC):
    X,y = createRandomLinear(samples_qty)
    Xs=np.c_[np.asarray(range(0,20+1,1))/10]


# closedForms(X,y)

# batchGradientDescent(X,y)

# stochastidGradientDescent(X,y)
X_vec=np.c_[np.ones((len(X),1)),X]
eta=0.007
n_epochs=500
t0, t1= 5,50
m=samples_qty


def learning_schedule(t):
    return t0/(t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(int(m/10)):          #uwaga, sample sa losowane (nie trzeba liczyc wszystkich na raz). moglibysmy losowa mniejsza ilosc razy
        random_index = np.random.randint(m)
        xi = X_vec[random_index:random_index+1]     
        yi= y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch*m +i)             #eta_min = 5/(50+50*100)= 
        theta=theta-eta * gradients

print(theta)
print('kot)')




#%%
for t in tlist:
