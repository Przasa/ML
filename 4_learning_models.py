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

def calc_theta_svd(X,y): #SVD = Singular Value Decompositino (do wyliczania macierzy pseudoodwrotnej)
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


def closedForms(X,y):
    theta_cform, theta_linreg, theta_svd = calc_theta_corm(X,y),calc_theta_linreg(X,y),calc_theta_svd(X,y)
    yp_cform,yp_linreg,yp_svd = predict_by_theta(Xs,theta_cform),predict_by_theta(Xs,theta_linreg),predict_by_theta(Xs,theta_svd)
    
    makeAnalyze()       #troche przerost formy z ta funckja

#TODO: zrob tabliczke z uzyskanyi wynikami gradientow i cech (dla wszystkich)
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
        gradients = 2/m + X_vec.T.dot(X_vec.dot(theta)-y)   #za kazdym razem, liczymu gradient dla wszystkich sampli
        theta = theta - eta*gradients
        glist.append(gradients.T[0])
        tlist.append(theta.T[0])
        tarr=np.asarray(tlist)
        garr=np.asarray(glist)
        if np.abs(gradients[0])< 0.001 and np.abs(gradients[1]) < 0.001:
            break

    plt.figure(1)
    plt.plot(tarr[:,0],tarr[:,1],'b')
    plt.xlabel('$x_0$')
    plt.ylabel("$x_1$")
    plt.grid(True)

    plt.figure(2)
    plt.plot(garr[:,0],garr[:,1],'b')
    plt.xlabel("gradient($x_0$)")
    plt.ylabel("graident($x_1$)")
    plt.grid(True)
    
def stochastidGradientDescent(X,y):
    X_vec=np.c_[np.ones((len(X),1)),X]
    eta=0.007
    n_epochs=50
    t0, t1= 5,50
    # m=samples_qty
    m=len(X_vec)

    def learning_schedule(t):
        return t0/(t+t1)

    theta = np.random.randn(2,1)

    glist=[]
    tlist=[]
    for epoch in range(n_epochs):
        for i in range(int(m)):          #uwaga, sample sa losowane (nie trzeba liczyc wszystkich na raz). moglibysmy losowa mniejsza ilosc razy
            random_index = np.random.randint(m)
            xi = X_vec[random_index:random_index+1]     
            yi= y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta)-yi)  #liczymy gradient dla jednego losowego sampla i mozemy przerwac liczenie w dowolnym momencie
            eta = learning_schedule(epoch*m +i)             #eta_min = 5/(50+50*100)= 
            theta=theta-eta * gradients
            glist.append([gradients[0][0],gradients[1][0]])
            tlist.append([theta[0][0],theta[1][0]])
    garr=np.asarray(glist)
    tarr=np.asarray(tlist)


    plt.figure(3)
    plt.plot(tarr[:,0],tarr[:,1],'b')
    plt.xlabel('$x_0$')
    plt.ylabel("$x_1$")
    plt.grid(True)
    plt.show()

    plt.figure(4)
    plt.plot(garr[:,0],garr[:,1],'b')
    plt.xlabel("gradient($x_0$)")
    plt.ylabel("graident($x_1$)")
    plt.grid(True)


print("code prepared")

def minibatchGradientDescent(X,y):
    theta_path_mgd = []
    grad_path_mgd = []

    n_iterations = 50
    minibatch_size = 20
    m=len(X)
    X_vec=np.c_[np.ones((len(X),1)),X]

    np.random.seed(42)
    theta = np.random.randn(2,1)  # inicjalizacja losowa

    t0, t1 = 200, 1000
    def learning_schedule(t):
        return t0 / (t + t1)

    t = 0
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_vec_shuffled = X_vec[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            t += 1
            xi = X_vec_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size]
            gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(t)
            theta = theta - eta * gradients
            theta_path_mgd.append(theta)
            grad_path_mgd.append(gradients)

    theta_path_mgd = np.asarray(theta_path_mgd)
    grad_path_mgd = np.asarray(grad_path_mgd)

    plt.figure(5)
    plt.plot(theta_path_mgd[:,0],theta_path_mgd[:,1],'b')
    plt.xlabel('$x_0$')
    plt.ylabel("$x_1$")
    plt.grid(True)
    plt.show()

    plt.figure(6)
    plt.plot(grad_path_mgd[:,0],grad_path_mgd[:,1],'b')
    plt.xlabel("gradient($x_0$)")
    plt.ylabel("graident($x_1$)")
    plt.grid(True)



# %%
samples_qty=100
#TODO zrob cos takiego dla wielu cech, nie tylko jednej z zerową
if('X' not in globals() or 'y' not in globals() or FORCE_CALC):
    X,y = createRandomLinear(samples_qty)
    Xs=np.c_[np.asarray(range(0,20+1,1))/10]


# TODO: troche otworz ta funkcje
# closedForms(X,y)

#TODO: zrobi jakies porownanie szybkosci tych gradientow
# batchGradientDescent(X,y)
# stochastidGradientDescent(X,y)
# minibatchGradientDescent(X,y)
#TODO: mozne wrzucic jeszcze uczenie przez  SGD Regressor

del(X)
del(y)




#%% Polynomial regression
from sklearn.preprocessing import PolynomialFeatures

if('X' not in globals() or 'y' not in globals() or FORCE_CALC):
    n_samples =100
    X = 6 * np.random.rand(n_samples,1)-3
    y= 0.5 * X**2  +2 + np.random.randn(n_samples,1)


poly = PolynomialFeatures(degree=2,include_bias=False)
X_poly=poly.fit_transform(X)        # nie musielismy wrzucac X0 = 1
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)

theta = [x for x  in lin_reg.coef_[0]]
theta.insert(0,lin_reg.intercept_)
X_poly_voc = np.c_[np.ones((len(X),1)),X_poly]
yp=X_poly_voc.dot(theta)
yp.resize((len(yp),1))

plt.figure(7)
plt.plot(X, y, "r.")
plt.plot(X, yp, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.title("Polynomial regression")
# save_fig("r_4_12")
plt.show()

#razczej tak rob
# X_new=np.linspace(-3, 3, 100).reshape(100, 1)
# X_new_poly = poly_features.transform(X_new)
# y_new = lin_reg.predict(X_new_poly)

