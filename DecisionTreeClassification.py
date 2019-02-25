#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
data=pd.read_csv("Data\data.csv")
data.info()

#%%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.diagnosis=[ 1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
xData=data.drop(["diagnosis"],axis=1)
#%%
x=(xData-np.min(xData))/(np.max(xData)-np.min(xData))
#%%
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.3,random_state=1)
#%%
from sklearn.tree import DecisionTreeClassifier
DTC=DecisionTreeClassifier()
DTC.fit(xTrain,yTrain)
print("Score Values: ",DTC.score(xTest,yTest))

#%% Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(n_estimators=50,random_state=1)
RFC.fit(xTrain,yTrain)
print("Score Values: ",RFC.score(xTest,yTest))