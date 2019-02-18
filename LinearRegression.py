#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#y=b0+b0*x1
#%%
# Linear Regression
df=pd.read_csv("Data\linear-regression-dataset.csv",sep=";")
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()
#%%
from sklearn.linear_model import LinearRegression
linearReg=LinearRegression()
#Eğer dataframe de veri uzunluğu(20,) gibi gösteriyorsa
#df.sutunadı.values.reshape(-1,1) yaparak düzeltebilirsin
x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)
linearReg.fit(x,y)
array=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
#Prediction
pre=linearReg.predict(array)
print(pre)
print("---------")
#x=0 ın kestiği nokta
pre1=linearReg.intercept_
print(pre1)
print("--------")
#Eğim
pre2=linearReg.coef_
print(pre2)
#%%
plt.scatter(x,y)
plt.show()
yhead=linearReg.predict(array)
plt.plot(array)