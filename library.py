#IV. SVM

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVR

#NOTE: maszyny wektorow nosnych (support vector machine) nadaja sie do klasfikacji liniowej, nieliniowej, regresji, wykr el. odstajacych
# margines nie zalezy od kolejnych przykladow ale przykladow na koncach ulicy (support vectors)

1) zwyczajne uczenie
svm_clf = SVC(kernel="linear", C=float("inf")) # from sklearn.svm import SVC
svm_clf = LinearSVC(C=1,loss='hinge')
#NOTE: uzywanie Linear SVC wymaga wczesniejszego skalowania (StandardScaler())
#NOTE: im wiekszy C, tym margines bardziej miekki
#NOTE: mozesz tez uzyc Stochastic Gradient Descent (SGDClassifier) (jak?) 
#NOTE: SVC obsluguje sztuczke z jadrem, swietny przy niewielkiej ilosci przykladow, ale niewydajny przy duzych: O(m^3xn), LinearSVC: O(mxn)
#NOTE: wyciaganie granicy decyzyjnej (do wykresow np.)
#      w = svm_clf.coef_[0], b = svm_clf.intercept_[0] w0*x0 + w1*x1 + b = 0  => x1 = -w0/w1 * x0 - b/w1

2) skalowanie danych wielomianowych
a) reczne dodawnia (dla linearSVC): PolynomialFeatures(degree=3)
b) wpudowane w jadra: SVC(kernel='poly',degree=3,coef0=1)
#NOTE: coef0 reguloje proporcje duzych wielomianow do malych 

3) Cechy podobienstwa: Radial Basis Function (RBF) => liczymy gausa do prawdopodobienstwa
SVC(kernel='rbf',gamma=5, C=0.001) 
#NOTE: gamma ustala szerokosc dzwonu
#NOTE: zwykle ustala sie punkt char. dla kazdego przykladu, ale kiedy jest ich duzo bedzie ciezko.
#NOTE: fi(x_v,l)=exp(-gamma*||x_v-l||^2) #l=punkt charakterystyczny

4) Regresja: odwrotne zadanie, chcemy miec najwiecej probek wewnatrz marginesow. 
svm_ref = LinearSVR(epsilon=1.5)
svm_clf = SVR(kernel='poly',degree=2, C=100, epsilon=0.1)
#NOTE: epsilon ustala szerokosc marginesow


