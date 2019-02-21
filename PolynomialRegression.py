#%%
#y=b0+b1*x+b2x^2+....
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
df=pd.read_csv("Data\polynomial-regression.csv",sep=";")
x=df.araba_fiyat.values.reshape(-1,1)
y=df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba fiyat")
plt.ylabel("hiz")
plt.show()
#%%
from sklearn.linear_model import LinearRegression
LinearReg=LinearRegression()
LinearReg.fit(x,y)
y_head=LinearReg.predict(x)
plt.plot(x,y_head,color="red",label="linear")
plt.show()
#%%
from sklearn.preprocessing import PolynomialFeatures
PolyReg =PolynomialFeatures(degree=2)
PolyReg.fit_transform(x)
#%%
from sklearn.linear_model import LinearRegression
LinearReg2=LinearRegression()
LinearReg2.fit(PolyReg,y)

y_head2=LinearReg2.predict(PolyReg)
plt.plot(x,y_head2,color="green",label="poly")
plt.legend()
plt.show()
