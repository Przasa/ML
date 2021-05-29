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

from os import pipe
from numpy.linalg.linalg import cholesky
from sklearn.utils.extmath import randomized_range_finder
import set_workspace as wspace
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
import scipy.linalg
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


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

    #liczenia reczne z theta (ale wyjeta z lin_rega)
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

    #liczenie dla wielomianow
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


def regulatedModels(X,Xs,X_poly,Xs_poly,y):
    global FIGURE
    # ‘auto’ chooses the solver automatically based on the type of data.
    # ‘svd’ uses a Singular Value Decomposition of X to compute the Ridge coefficients. More stable for singular matrices than ‘cholesky’.
    # ‘cholesky’ uses the standard scipy.linalg.solve function to obtain a closed-form solution.
    # ‘sparse_cg’ uses the conjugate gradient solver as found in scipy.sparse.linalg.cg. As an iterative algorithm, this solver is more appropriate than ‘cholesky’ for large-scale data (possibility to set tol and max_iter).
    # ‘lsqr’ uses the dedicated regularized least-squares routine scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative procedure.
    # ‘sag’ uses a Stochastic Average Gradient descent, and ‘saga’ uses its improved, unbiased version named SAGA. Both methods also use an iterative procedure, and are often faster than other solvers when both n_samples and n_features are large. Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.
    solvers = ['auto','svd','cholesky','sparse_cg','lsqr','sag']
    colors = ['green','red','black','orange','grey','pink']
    alphas=[1e-5,0.1,0.5,0.9,1]


    fig,(ax1,ax2)= plt.subplots(1,2)
    fig.suptitle('porownanie roznych modeli z kara L2 (Ridge)')
    fig.set_size_inches(12, 6)
    fig.number=FIGURE; FIGURE+=1

    #porownanie dla modeli linionwych
    ax1.set_title('Predykcja dla linear')
    ax1.plot(X,y,'b.',label='samples')
    for slv,clr in zip(solvers,colors):
        model = Ridge(alpha=1,solver=slv,random_state=42,tol=1e-3)
        model.fit(X,y)
        predictions = model.predict(Xs)
        ax1.plot(Xs,predictions,clr,label=slv)
    sgd_model = SGDRegressor(penalty='l2')
    sgd_model.fit(X,y)
    ax1.plot(Xs,sgd_model.predict(Xs),'magenta',label='SGD_reg')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # porownanie dla modeli wielomianowych
    ax2.set_title('Predykcja dla polynomial')
    ax2.plot(X,y,'b.',label='samples')
    for slv,clr in zip(solvers,colors):
        model = Ridge(alpha=1,solver=slv,random_state=42,tol=1e-3)
        model.fit(X_poly,y)
        predictions = model.predict(Xs_poly)
        ax2.plot(Xs,predictions,clr,label=slv)
    sgd_model = SGDRegressor(penalty='l2')
    sgd_model.fit(X_poly,y)
    ax2.plot(Xs,sgd_model.predict(Xs_poly),'magenta',label='SGD_reg')
    ax2.axis([0,3,0,4])
    plt.legend()
    ax2.grid(True)


    #porownanie dla roznych kary alpha (modele liniowe)
    fig,(ax1,ax2)= plt.subplots(1,2)
    fig.suptitle('porownanie modelu z kara L2, dla roznych alpha')
    fig.set_size_inches(12, 6)
    fig.number=FIGURE; FIGURE+=1
    ax1.set_title('Predykcja z Ridge (cholesky[=def])\n(porownanie a dla linear)')
    ax1.plot(X,y,'b.',label='samples')
    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha,solver='cholesky',random_state=42)
        ridge_model.fit(X,y)
        yp=ridge_model.predict(Xs)
        ax1.plot(Xs,yp,label='alpha='+str(alpha))
    ax1.grid(True)
    ax1.axis([0.5,1.5,1,2])       # TODO: wyrkres bez zooma
    ax1.legend()

    #porownanie dla roznych alpha (modele wielomianowe)
    ax2.set_title('Predykcja z Ridge (cholesky[=def=cl.form])\n(porownanie a dla polynomials)')
    ax2.plot(X,y,'b.',label='samples')
    for alpha in alphas:
        pipeline = Pipeline([
            ('Polynomial',PolynomialFeatures(degree=3,include_bias=False)),
            ('Scaler',StandardScaler()),
            ('Ridge',Ridge(alpha=alpha,solver='cholesky',random_state=42))
        ])
        pipeline.fit(X,y)
        yp=pipeline.predict(Xs)
        ax2.plot(Xs,yp,label='alpha='+str(alpha))
    ax2.grid(True)
    ax2.axis([0.5,1.5,1,2])       # TODO: wyrkres bez zooma
    ax2.legend()

    fig,(ax1,ax2)= plt.subplots(1,2)
    fig.number=FIGURE; FIGURE+=1
    fig.suptitle('porownanie modelu z kara L1 (Lasso), dla roznych alpha')
    fig.set_size_inches(12, 6)

    #lasso jest modelem liniowym. liczy wiec prawdopodobnie z closed form
    ax1.set_title('model liniowy')
    ax1.plot(X,y,'b.',label='samples')
    for alpha in alphas:
        lasso_model=Lasso(alpha=alpha)  
        lasso_model.fit(X,y)
        ypred=lasso_model.predict(Xs)
        ax1.plot(Xs,ypred,label=alpha)
    ax1.grid(True)
    ax1.axis([1,2,1,2])
    ax1.legend()

    ax2.set_title('model wielomianowy')
    ax2.plot(X,y,'b.',label='samples')
    for alpha in alphas:
        lasso_model=Lasso(alpha=alpha)  
        lasso_model.fit(X_poly,y)
        ypred=lasso_model.predict(Xs_poly)
        ax2.plot(Xs,ypred,label=alpha)
    ax2.grid(True)
    ax2.axis([1,2,1,2])
    ax2.legend()


    fig,(ax1,ax2)= plt.subplots(1,2)
    fig.number=FIGURE; FIGURE+=1
    fig.suptitle('porownanie modelu z kara L1 i L2 (ElasticNet), dla roznych alpha')
    fig.set_size_inches(12, 6)

    #lasso jest modelem liniowym. liczy wiec prawdopodobnie z closed form
    ax1.set_title('model liniowy')
    ax1.plot(X,y,'b.',label='samples')
    for alpha in alphas:
        elnet_model=ElasticNet(alpha=alpha,l1_ratio=0.5, random_state=42)  
        elnet_model.fit(X,y)
        ypred=elnet_model.predict(Xs)
        ax1.plot(Xs,ypred,label=alpha)
    ax1.grid(True)
    ax1.axis([1,2,1,2])
    ax1.legend()

    ax2.set_title('model wielomianowy')
    ax2.plot(X,y,'b.',label='samples')
    for alpha in alphas:
        elnet_model=ElasticNet(alpha=alpha,l1_ratio=0.5, random_state=42)  
        elnet_model.fit(X_poly,y)
        ypred=elnet_model.predict(Xs_poly)
        ax2.plot(Xs,ypred,label=alpha)
    ax2.grid(True)
    ax2.axis([1,2,1,2])
    ax2.legend()

def early_stopping(X,y):
    global FIGURE
    X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10) #dlaczego tylko polowa?
    pipeline = Pipeline([
        ('poly_feature',PolynomialFeatures(degree=90,include_bias=False)), #90stopni!
        ("standard_scaler",StandardScaler())
    ])
    X_train_poly = pipeline.fit_transform(X_train)
    X_val_poly = pipeline.fit_transform(X_val)

    sgd_model = SGDRegressor(max_iter=1,tol=-np.infty,warm_start=True,penalty=None, learning_rate ="constant",eta0=0.0005, random_state=42)

    min_error = float("inf")
    best_epoch =None
    best_model = None
    epoch_lst,error_lst = [],[]
    for epoch in range(1000):
        sgd_model.fit(X_train_poly,y_train)
        ypred = sgd_model.predict(X_val_poly)
        error = mean_squared_error(y_val,ypred)
        epoch_lst.append(epoch)
        error_lst.append(error)
        if error< min_error:
            best_epoch = epoch
            best_model = clone(sgd_model)
            min_error=error

    plt.figure(FIGURE); FIGURE+-1
    plt.title('early stopping (for SGDregressor)')
    plt.plot(epoch_lst,error_lst,label='failures')
    plt.plot([0,1000],[min_error,min_error],'b--',label='min error='+str(min_error))
    # plt.text(10,10,('min_error='),bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5},fontsize='large')
    plt.axis([np.max([np.argmin(error_lst)-300,0]),np.min([1000,np.argmin(error_lst)+100]),0.9*min_error,1.1*min_error])
    plt.legend()
    plt.grid(True)

# %%
samples_qty=100
#TODO zrob cos takiego dla wielu cech, nie tylko jednej z zerową
if('X' not in globals() or 'y' not in globals() or FORCE_CALC):
    X,y = createRandomLinear(samples_qty)
    Xs=np.c_[np.asarray(range(0,20+1,1))/10]


#TODO: troche otworz closedForms (przerost formy nad trescia)
#TODO: mozne wrzucic jeszcze uczenie przez  SGD Regressor (tzn z funckji SGDRegressor, a nie z reki)
#TODO: te wykresy mozna naprawci troche
#TODO: zrobi jakies porownanie szybkosci tych gradientow (przy tych samych wejsciach)
# closedForms(X,y)
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

del(X); del(y)
if ('X' not in globals() or 'y' not in globals()):
    X= 3*np.random.rand(samples_qty,1)
    y= 1 + 0.5*X + np.random.randn(samples_qty,1)
    Xs = np.c_[range(0,30+1,1)]; Xs=Xs/10

    polfeat = PolynomialFeatures(degree=3,include_bias=False)
    X_poly = polfeat.fit_transform(X)
    Xs_poly = polfeat.fit_transform(Xs)

# regulatedModels(X,Xs,X_poly,Xs_poly,y)

#wczesne zatrzymywanie (to tez regulacja_).
del(X); del(y)
if ('X' not in globals() or 'y' not in globals()):
    samples_qty = 100
    np.random.seed(42)
    X = 6 * np.random.rand(samples_qty, 1) - 3
    y = 2 + X + 0.5 * X**2 + np.random.randn(samples_qty, 1)

early_stopping(X,y)

#graf 

# %% regresja logistyczne
#przyjzyj sie tym obliczeniom jeszcze

from sklearn.linear_model import LogisticRegression

if 'iris' not in globals():
    from sklearn import datasets
    iris = datasets.load_iris()
    list(iris.keys())


#licznie dla jednej cechy
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
X=iris['data'][:,3:]      #dlugosc platka
y=(iris['target']==2).astype(np.int)        #virgnica
log_reg.fit(X,y)
X_samples=np.linspace(0,3,1000).reshape(-1, 1)
preds = log_reg.predict_proba(X_samples)        #oblicza tez przyporzadkowanie negatywne
decision_boundary = X_samples[preds[:, 1] >= 0.5][0]

plt.plot(X_samples, preds[:, 1], "g-", linewidth=2, label="Iris virginica")
plt.plot(X_samples, preds[:, 0], "b--", linewidth=2, label="Pozostałe")
plt.xlabel("Szerokość płatka (cm)", fontsize=14)
plt.ylabel("Prawdopodobieństwo", fontsize=14)
plt.legend()
plt.grid(True)


#liczenie dla 2 cech
from sklearn.linear_model import LogisticRegression

X = iris["data"][:, (2, 3)]  # długość płatka, szerokość płatka
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression(solver="lbfgs", C=10**10, random_state=42)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Pozostałe", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Długość płatka", fontsize=14)
plt.ylabel("Szerokość płatka", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
# save_fig("r_4_24")
plt.show()


#softmax
X = iris["data"][:, (2, 3)]  # długość płatka, szerokość płatka
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X, y)
x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]


y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Długość płatka", fontsize=14)
plt.ylabel("Szerokość płatka", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
# save_fig("r_4_25")
plt.show()














