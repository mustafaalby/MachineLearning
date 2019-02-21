#%%
import pandas as pd
import matplotlib.pyplot as plt
#%% Data Tanıma
data=pd.read_csv("Data\data.csv")
data.head()
#%%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.head()

#%%
M=data[data.diagnosis=="M"]#data içerisindeki M olan satırları M değişkenine at
B=data[data.diagnosis=="B"]
#%%
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Kotu")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="IYI")
plt.legend()
plt.xlabel("radiusMean")
plt.ylabel("textureMean")
plt.show()
#%% KNN
data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
xData=data.drop(["diagnosis"],axis=1)
#%% Normalization
###Önemli
x=(xData-np.min(xData))/(np.max(xData)-np.min(xData))
###Önemli
#%%
from sklearn.model_selection import train_test_split
xTrain,XTest,YTrain,YTest=train_test_split(x,y,test_size=0.3,random_state=1)
#%%
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(xTrain,YTrain)
prediction=knn.predict(XTest)
#%%
print("3 nn score: ",(knn.score(XTest,YTest)))
#%%K Degerini bul
list=[]
for each in range(1,10):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(xTrain,YTrain)
    list.append(knn2.score(XTest,YTest))

plt.plot(range(1,10),list)
plt.xlabel("K Degerleri")
plt.ylabel("accuary")
plt.show()

