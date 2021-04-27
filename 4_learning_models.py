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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#%% funckje
FIGURE=1

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
    global FIGURE
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

    plt.figure(FIGURE); FIGURE+=1
    plt.plot(tarr[:,0],tarr[:,1],'b')
    plt.xlabel('$x_0$')
    plt.ylabel("$x_1$")
    plt.grid(True)

    plt.figure(FIGURE) ; FIGURE+=1
    plt.plot(garr[:,0],garr[:,1],'b')
    plt.xlabel("gradient($x_0$)")
    plt.ylabel("graident($x_1$)")
    plt.grid(True)
    
def stochastidGradientDescent(X,y):
    global FIGURE
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


    plt.figure(FIGURE);FIGURE+=1 
    plt.plot(tarr[:,0],tarr[:,1],'b')
    plt.xlabel('$x_0$')
    plt.ylabel("$x_1$")
    plt.grid(True)
    plt.show()

    plt.figure(FIGURE);FIGURE+=1
    plt.plot(garr[:,0],garr[:,1],'b')
    plt.xlabel("gradient($x_0$)")
    plt.ylabel("graident($x_1$)")
    plt.grid(True)


print("code prepared")

def minibatchGradientDescent(X,y):
    global FIGURE
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

    plt.figure(FIGURE); FIGURE+=1
    plt.plot(theta_path_mgd[:,0],theta_path_mgd[:,1],'b')
    plt.xlabel('$x_0$')
    plt.ylabel("$x_1$")
    plt.grid(True)
    plt.show()

    plt.figure(FIGURE); FIGURE+=1
    plt.plot(grad_path_mgd[:,0],grad_path_mgd[:,1],'b')
    plt.xlabel("gradient($x_0$)")
    plt.ylabel("graident($x_1$)")
    plt.grid(True)

def basic_poly_reg(X,y):
    global FIGURE
    poly = PolynomialFeatures(degree=2,include_bias=False)
    X_poly=poly.fit_transform(X)        # nie musielismy wrzucac X0 = 1
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly,y)

    #liczenia reczne z theta
    theta = [x for x  in lin_reg.coef_[0]]
    theta.insert(0,lin_reg.intercept_)
    X_poly_vec = np.c_[np.ones((len(X),1)),X_poly]
    yp=X_poly_vec.dot(theta)
    yp.resize((len(yp),1))

    #liczenie automatyczne z funkcji predict
    Xs=np.linspace(-3, 3, 100).reshape(100, 1)
    Xs_poly = poly.transform(Xs)
    ysp = lin_reg.predict(Xs_poly)

    plt.figure(FIGURE); FIGURE+=1
    plt.plot(X, y, "r.",label='samples')
    plt.plot(X, yp, "b.",label='predictions (by theta)')
    plt.plot(Xs, ysp, "g",label='predictions (by predict())')
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([-3, 3, 0, 10])
    plt.legend()
    plt.title("Polynomials and LinearRegression")
    # save_fig("r_4_12")
    plt.show()

    plt.figure(FIGURE); FIGURE+=1
    swd_conf = (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1),("black",2,5))   #style width degree
    for style, width, degree in swd_conf:
        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
        std_scaler = StandardScaler()
        lin_reg = LinearRegression()
        polynomial_regression = Pipeline([
                ("poly_features", polybig_features),
                ("std_scaler", std_scaler),
                ("lin_reg", lin_reg),
            ])
        polynomial_regression.fit(X, y)
        ysp_multi = polynomial_regression.predict(Xs)
        plt.plot(Xs, ysp_multi, style, label=str(degree), linewidth=width)

    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.title("Polynomials comparision\nCalculated by predict()")
    plt.axis([-3, 3, 0, 10])
    # save_fig("r_4_14")
    plt.show()

def learning_curves(X,y):

    def plot_learning_curves(model, X, y,header):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
        train_errors, val_errors = [], []
        for m in range(1, len(X_train)):        #zmienia sie dlugosc zbiory uczacego, ale walidacyjne nie (walidacyjny jest natomiast zalezny od modelu trenowanego na zbierze uczacego)
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])    # zmianie sie dlugosc przewidywanego zbioru uczecego
            y_val_predict = model.predict(X_val)            # a tu dlugosc sie nie zmienia
            train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
            val_errors.append(mean_squared_error(y_val, y_val_predict))

        global FIGURE
        plt.figure(FIGURE); FIGURE+=1
        plt.title('Krzywe Bledu ('+header+')')
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Zestaw uczący")
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Zestaw walidacyjny")
        plt.legend(loc="upper right", fontsize=14)          # nieukazane w książce
        plt.xlabel("Rozmiar zestawu uczącego", fontsize=14) # nieukazane
        plt.ylabel("Błąd RMSE", fontsize=14)                # nieukazane
        plt.text(20,train_errors[-1]+0.5,('RMSE_train='+str(train_errors[-1])+'\nRMSE_val='+str(val_errors[-1])),bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},fontsize='large')
        plt.grid(True)
        

    linear_model= LinearRegression()
    polynomial_model_2 = Pipeline([
        # ('Standard Scaler',StandardScaler()),
        ('Polynomialer', PolynomialFeatures(degree=2,include_bias=False)),
        ("Linear Regression",LinearRegression())
    ])
    polynomial_model_10 = Pipeline([
        # ('Standard Scaler',StandardScaler()),
        ('Polynomialer', PolynomialFeatures(degree=10,include_bias=False)),
        ("Linear Regression",LinearRegression())
    ])

    plot_learning_curves(linear_model,X,y,header='linear')
    plot_learning_curves(polynomial_model_2,X,y,header='polynomial (2)')
    plot_learning_curves(polynomial_model_10,X,y,header='polynomial (10)')


# %%
samples_qty=100
#TODO zrob cos takiego dla wielu cech, nie tylko jednej z zerową
if('X' not in globals() or 'y' not in globals() or FORCE_CALC):
    X,y = createRandomLinear(samples_qty)
    Xs=np.c_[np.asarray(range(0,20+1,1))/10]


# TODO: troche otworz ta funkcje (przerost formy nad trescia)
# closedForms(X,y)
#TODO: mozne wrzucic jeszcze uczenie przez  SGD Regressor (tzn z funckji SGDRegressor, a nie z reki)
#TODO: te wykresy mozna naprawci troche
#TODO: zrobi jakies porownanie szybkosci tych gradientow (przy tych samych wejsciach)
# batchGradientDescent(X,y)
# stochastidGradientDescent(X,y)
# minibatchGradientDescent(X,y)

del(X);del(y)
if('X' not in globals() or 'y' not in globals() or FORCE_CALC):
    n_samples =100
    X = 6 * np.random.rand(n_samples,1)-3
    y= 0.5 * X**2  +2 + np.random.randn(n_samples,1)

# basic_poly_reg(X,y)
# learning_curves(X,y)

#%% Regularyzowane modele


