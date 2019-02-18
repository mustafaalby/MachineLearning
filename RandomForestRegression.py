
#%%
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
df=pd.read_csv("Data\decision-tree-regression-dataset.csv",sep=";",header=None)
x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)
#%%
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)

Xx=np.arange(min(x),max(x),0.01).reshape(-1,1)
yHead=rf.predict(Xx)

plt.scatter(x,y,color="red")
plt.plot(Xx,yHead,color="green")
plt.xlabel("level")
plt.ylabel("ucret")
plt.show()
