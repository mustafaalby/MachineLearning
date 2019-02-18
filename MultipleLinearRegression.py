#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#y=b0+b1*x1+b2*x2.b3*x3...
#%%
df=pd.read_csv("Data\multiple.csv",sep=";")
x=df.iloc[:,[0,2]].values
y=df.maas.values.reshape(-1,1)
#%%
multiple_linReg=LinearRegression()
multiple_linReg.fit(x,y)
print("0. noktasındaki değer",multiple_linReg.intercept_)
print("Eğim",multiple_linReg.coef_)

#%%
#predict
print("predict",multiple_linReg.predict(np.array([[10,35],[5,35]])))
