import numpy as np
from sklearn.model_selection import StratifiedKFold


# X, y = np.ones((50, 1)), np.hstack(([2] * 45, [3] * 5))
# print(X,y)
# skf = StratifiedKFold(n_splits=2)
# for train, test in skf.split(X, y):
#     print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))


# variables = 60000
# X=np.asarray([[x,x+10] for x in range(variables)])
# y=np.asarray([x+30 for x in range(variables)])
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[10, 11], [13, 14]])
y = np.array([[1], [1], [1], [1],[1],[1]])
print(X,y)
skf = StratifiedKFold(n_splits=6,shuffle=True)


print(skf.get_n_splits(X, y))
print(skf)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]



# a=np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]])
# b=np.array([1,2,3,4,5,6,7,8,9]) 
# # print(a)
# # print(b)

# sfold=StratifiedKFold(n_splits=3)
# for train_index, test_index in sfold.split(a,b):
#     print(a[train_index],a[test_index])
#     print(b[train_index],b[test_index])


