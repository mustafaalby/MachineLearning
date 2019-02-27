#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
#%%
#Normalization
x=(x-np.min(x))/(np.max(x)-np.min(x))
x.dtype
#%%

from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.25)
#%%
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=3)
#%%
from sklearn.model_selection import cross_val_score
accurisies=cross_val_score(estimator=knn,X=xTrain,y=yTrain,cv=10)
print("avarage accurisies ",np.mean(accurisies))
print("avarage std ",np.std(accurisies))
#%%
KNN.fit(xTrain,yTrain)
print("test accuary ",KNN.score(xTest,yTest))
#%%
#Grid search cross validation
from sklearn.model_selection import GridSearchCV
grid={"n_neighbors":np.arange(1,50)}
knn=KNeighborsClassifier()

knnCV=GridSearchCV(knn,grid,cv=10)
knnCV.fit(x,y)


#%%
print("tuned hypermeter K: ",knnCV.best_params_ )
print("tuned parametreye göre accuary(Score): ",knnCV.best_score_ )
#%%
x=x[:100,:]
y=y[:100]
from sklearn.linear_model import LogisticRegression
grid ={"C": np.logspace(-3,3,7),"penalty":["11","12"]}
logreg=LogisticRegression()
logregCV=GridSearchCV(logreg,grid,cv=10)
print("tuned best parameters ",logregCV.best_params_)
print("accuary ",logregCV.best_score_)