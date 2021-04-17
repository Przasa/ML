# %% initials
FORCE_CALC=True

import set_workspace as wspace
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.linalg
# %%
l=100
#TODO zrob cos takiego dla wielu cech, nie tylko jednej z zerową
if('X' not in globals()or FORCE_CALC):
    X = 2 * np.random.rand(l, 1)        # nie myl X z calym zestawem cech Xn. X to tylko jedna cecha (tj X0=1, X1=X)
    y = 15 + 3 * X + np.random.randn(l, 1) #randn to szum gausowski
    Xn=np.c_[np.ones((l,1)),X]
    theta_cform=np.linalg.inv(Xn.T.dot(Xn)).dot(Xn.T).dot(y)

    Xs=np.asarray(range(0,20+1,1))/10
    Xs.resize(len(Xs),1)
    Xsb=np.c_[np.ones((len(Xs),1)),Xs]
    yp_cform=Xsb.dot(theta_cform)

lin_reg=LinearRegression()
lin_reg.fit(X=Xn,y=y)
yp_linreg=lin_reg.predict(Xn)
theta_linreg=np.r_[[lin_reg.intercept_,lin_reg.coef_[0][1:]]]

theta_svd =LinearRegression()
lin_reg.fit(X=Xn,y=y)
yp_linreg=lin_reg.predict(Xn)
# print("")
# print(lin_reg.intercept_, lin_reg.coef_)

plt.figure(1)
plt.plot(X,y,'b.',label='samples')
plt.plot(Xs,yp_cform,'r-',label='prediction (closed form)')
plt.plot(X,yp_linreg,'g-',label='prediction (linear reg)')
plt.grid(True)
plt.title(label='1: prediction')
plt.legend(loc='upper left')
# %%
# %%
plt.show()

plt.figure(2)
plt.plot(X,y,'b.',label='samples')
plt.plot(Xs,yp_cform,'r-',label='prediction (closed form)')
plt.plot(X,yp_linreg,'g-',label='prediction (linear reg)')
plt.axis([1,1.1,17,19])
plt.grid(True)
plt.title(label='2: prediction (zoomed.)')
plt.legend(loc='upper left')
plt.show()




# %%

# Xs=     (x0=1,  x1=X)
#         1     0
#         1     0.1
#         1     0.2
#         1     0.3
#         1     ...
#         1     2
# theta_cform = a0=15  
#         a1=2.8 
# y(x)  Xs*theta_cform=a0*x0+a1*x1 =a0 + a1*X
# y(0)          =a0*1+a1*0
# y(1.1)        =a0*1+a1*1.1


# plt.show()
# wspace.save_fig('0_init')


# %% ktos to lubi to i owo. 

np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
np.c_[np.ones((100, 1)), X]
X_new_b.dot(theta_best)

from sklearn.linear_model import LinearRegression
lin_reg.intercept_, lin_reg.coef_

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd

#wsadowiec

