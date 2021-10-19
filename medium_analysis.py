#!/usr/bin/env python
# coding: utf-8
#In[1]
import pandas as pd
#%%
import matplotlib.pyplot as plt
#%%
import numpy as np
#%%
from sklearn.linear_model import LinearRegression
#%%
from sklearn import datasets, linear_model
#%%
from sklearn.metrics import *
#%%
import statistics
#%%
#Import data:
df = pd.read_csv("df.csv")
#Preview data:
print(df.head(10))
# %%
import seaborn as sns
# %%
import matplotlib.pyplot as plt
# %%
sns.regplot(x=df["BMXWT"], y=df["BMXHIP"], line_kws={"color":"r","alpha":0.7,"lw":5})
plt.show()
# %%
sns.regplot(x=df["BMXHT"], y=df["BMXHIP"], line_kws={"color":"r","alpha":0.7,"lw":5})
plt.show()
# %%
sns.regplot(x=df["BMXWAIST"], y=df["BMXHIP"], line_kws={"color":"r","alpha":0.7,"lw":5})
plt.show()
# %%
sns.regplot(x=df["BMXWT"], y=df["BMXHIP"], fit_reg=False)
sns.regplot(x=df["BMXHT"], y=df["BMXHIP"], fit_reg=False)
sns.regplot(x=df["BMXWAIST"], y=df["BMXHIP"], fit_reg=False)
# %%
#Pearson correlation:
from numpy.random import randn
# %%
from numpy.random import seed
# %%
from scipy.stats import pearsonr
# %%
#CC between Hip circumference and weight:
corr, _ = pearsonr(df["BMXHIP"], df["BMXWT"])
print('Pearsons correlation: %.3f' % corr)
# %%
#CC between Hip circumference and height:
corr, _ = pearsonr(df["BMXHIP"], df["BMXHT"])
print('Pearsons correlation: %.3f' % corr)
# %%
#CC between Hip circumference and Waist circumference:
corr, _ = pearsonr(df["BMXHIP"], df["BMXWAIST"])
print('Pearsons correlation: %.3f' % corr)
#%%
#Regression Models:
y = df['BMXHIP']
#%%
#Model 1
X1 = df[['BMXHT']]
# %%
model_1 = linear_model.LinearRegression()
model_1.fit(X1, y)
# %%
print(model_1.coef_)
print(model_1.intercept_)
#%%
r_sq_model_1 = model_1.score(X1, y)
print(r_sq_model_1)
#%%
y1_pred = model_1.predict(X1)
print('predicted response:', y1_pred, sep='\n')
#%%
import numpy as np
#%%
import statsmodels.api as sm
#%%
#Add a constant:
X1a = sm.add_constant(X1)
model_1a = sm.OLS(y, X1a)
#%%
results_model_1a = model_1a.fit()
print(results_model_1a.summary())
#%%
#Residual Standard Error (RSE):
model_1a = sm.OLS(y, X1a).fit()
model_1a.resid.std(ddof=X1a.shape[1])
#%%
#Bland-Altman Plot:
df['BMXHIP_1'] = 78.90301 + (0.16066*df['BMXHT'])
df['md_1'] = df['BMXHIP']-df['BMXHIP_1']
print(df.head(10))
#%%
df['mean_BMXHIP'] = statistics.mean(df['BMXHIP'])
df['sd_BMXHIP'] = statistics.stdev(df['BMXHIP'])
df['mean_BMXHIP_1'] = statistics.mean(df['BMXHIP_1'])
df['sd_BMXHIP_1'] = statistics.stdev(df['BMXHIP_1'])
df['mean_md_1'] = statistics.mean(df['md_1'])
df['sd_md_1'] = statistics.stdev(df['md_1'])
#%%
#Plot drawn:
print("Mean md_1:", statistics.mean(df['md_1']))
print("SD md_1: ", statistics.stdev(df['md_1']))
print("LLA: ", statistics.mean(df['md_1'])-(1.96*statistics.stdev(df['md_1'])))
print("ULA: ", statistics.mean(df['md_1'])+(1.96*statistics.stdev(df['md_1'])))

plt.plot( 'BMXHIP', 'md_1', data=df, linestyle='none', marker='o')
plt.axhline(y = 0, color = 'black', linestyle = '-')
plt.axhline(y = -0.00043955585024463647, color = 'blue', linestyle = '--')
plt.axhline(y = -29.126151124880025, color = 'red', linestyle = '--')
plt.axhline(y = 29.125272013179533, color = 'red', linestyle = '--')
plt.show()
#%%
#Limits of Agreement amplitude:
LOAA_1 = 29.126151124880025 + 29.125272013179533
print("LOAA_1:" , LOAA_1)
# %%
#Model 2
X2 = df[['BMXWT']]
# %%
model_2 = linear_model.LinearRegression()
model_2.fit(X2, y)
# %%
print(model_2.coef_)
print(model_2.intercept_)
#%%
#Add a constant:
X2a = sm.add_constant(X2)
model_2a = sm.OLS(y, X2a)
#%%
results_model_2a = model_2a.fit()
#%%
print(results_model_2a.summary())
#%%
#Residual Standard Error (RSE):
model_2a = sm.OLS(y, X2a).fit()
model_2a.resid.std(ddof=X2a.shape[1])
#%%
#Bland-Altman Plot:
df['BMXHIP_2'] = 59.709755 + (0.568801*df['BMXWT'])
df['md_2'] = df['BMXHIP']-df['BMXHIP_2']
print(df.head(10))
#%%
df['mean_BMXHIP_2'] = statistics.mean(df['BMXHIP_2'])
df['sd_BMXHIP_2'] = statistics.stdev(df['BMXHIP_2'])
df['mean_md_2'] = statistics.mean(df['md_2'])
df['sd_md_2'] = statistics.stdev(df['md_2'])
print(df.head(10))
#%%
#Plot drawn:
print("Mean md_2:", statistics.mean(df['md_2']))
print("SD md_2: ", statistics.stdev(df['md_2']))
print("LLA: ", statistics.mean(df['md_2'])-(1.96*statistics.stdev(df['md_2'])))
print("ULA: ", statistics.mean(df['md_2'])+(1.96*statistics.stdev(df['md_2'])))

plt.plot( 'BMXHIP', 'md_2', data=df, linestyle='none', marker='o')
plt.axhline(y = 0, color = 'black', linestyle = '-')
plt.axhline(y = -6.812200491674594e-08, color = 'blue', linestyle = '-')
plt.axhline(y = -13.974429503341867, color = 'red', linestyle = '-')
plt.axhline(y = 13.974429367097859, color = 'red', linestyle = '-')
plt.show()
#%%
#Limits of Agreement amplitude:
LOAA_2 = 13.974429503341867 + 13.974429367097859
print("LOAA_2:" , LOAA_2)
# %%
#Model 3
X3 = df[['BMXWAIST']]
# %%
model_3 = linear_model.LinearRegression()
model_3.fit(X3, y)
# %%
print(model_3.coef_)
print(model_3.intercept_)
#%%
#Add a constant:
X3a = sm.add_constant(X3)
model_3a = sm.OLS(y, X3a)
#%%
results_model_3a = model_3a.fit()
print(results_model_3a.summary())
#%%
#Residual Standard Error (RSE):
model_3a = sm.OLS(y, X3a).fit()
model_3a.resid.std(ddof=X3a.shape[1])
#%%
#Bland-Altman Plot:
df['BMXHIP_3'] = 35.59334 + (0.71649*df['BMXWAIST'])
df['md_3'] = df['BMXHIP']-df['BMXHIP_3']
print(df.head(10))
#%%
df['mean_BMXHIP_3'] = statistics.mean(df['BMXHIP_3'])
df['sd_BMXHIP_3'] = statistics.stdev(df['BMXHIP_3'])
df['mean_md_3'] = statistics.mean(df['md_3'])
df['sd_md_3'] = statistics.stdev(df['md_3'])
print(df.head(10))
#%%
#Plot drawn:
print("Mean md_3:", statistics.mean(df['md_3']))
print("SD md_3: ", statistics.stdev(df['md_3']))
print("LLA: ", statistics.mean(df['md_3'])-(1.96*statistics.stdev(df['md_3'])))
print("ULA: ", statistics.mean(df['md_3'])+(1.96*statistics.stdev(df['md_3'])))

plt.plot( 'BMXHIP', 'md_3', data=df, linestyle='none', marker='o')
plt.axhline(y = 0, color = 'black', linestyle = '-')
plt.axhline(y = 0.00031253381618476223, color = 'blue', linestyle = '--')
plt.axhline(y = -14.118955041009063, color = 'red', linestyle = '--')
plt.axhline(y = 14.119580108641433, color = 'red', linestyle = '--')
plt.show()
#%%
#Limits of Agreement amplitude:
LOAA_3 = 14.118955041009063 + 14.119580108641433
print("LOAA_3:" , LOAA_3)
# %%
#Model 4
X4 = df[['BMXWT', 'BMXHT']]
# %%
model_4 = linear_model.LinearRegression()
model_4.fit(X4, y)
# %%
print(model_4.coef_)
print(model_4.intercept_)
#%%
#Add a constant:
X4a = sm.add_constant(X4)
model_4a = sm.OLS(y, X4a)
#%%
results_model_4a = model_4a.fit()
print(results_model_4a.summary())
#%%
#Residual Standard Error (RSE):
model_4a = sm.OLS(y, X4a).fit()
model_4a.resid.std(ddof=X4a.shape[1])
#%%
#Bland-Altman Plot:
df['BMXHIP_4'] = 137.920164 + (-0.518439*df['BMXHT']) + (0.669106*df['BMXWT'])
df['md_4'] = df['BMXHIP']-df['BMXHIP_4']
print(df.head(10))
#%%
df['mean_BMXHIP_4'] = statistics.mean(df['BMXHIP_4'])
df['sd_BMXHIP_4'] = statistics.stdev(df['BMXHIP_4'])
df['mean_md_4'] = statistics.mean(df['md_4'])
df['sd_md_4'] = statistics.stdev(df['md_4'])
print(df.head(10))
#%%
#Plot drawn:
print("Mean md_4:", statistics.mean(df['md_4']))
print("SD md_4: ", statistics.stdev(df['md_4']))
print("LLA: ", statistics.mean(df['md_4'])-(1.96*statistics.stdev(df['md_4'])))
print("ULA: ", statistics.mean(df['md_4'])+(1.96*statistics.stdev(df['md_4'])))

plt.plot( 'BMXHIP', 'md_4', data=df, linestyle='none', marker='o')
plt.axhline(y = 0, color = 'black', linestyle = '-')
plt.axhline(y = 0.0014407432010600173, color = 'blue', linestyle = '--')
plt.axhline(y = -10.529108193463623, color = 'red', linestyle = '--')
plt.axhline(y = 10.531989679865744, color = 'red', linestyle = '--')
plt.show()
#%%
#Limits of Agreement amplitude:
LOAA_4 = 10.529108193463623 + 10.531989679865744
print("LOAA_4:" , LOAA_4)
# %%
#Model 5
X5 = df[['BMXHT', 'BMXWAIST']]
# %%
model_5 = linear_model.LinearRegression()
model_5.fit(X5, y)
# %%
print(model_5.coef_)
print(model_5.intercept_)
#%%
#Add a constant:
X5a = sm.add_constant(X5)
model_5a = sm.OLS(y, X5a)
#%%
results_model_5a = model_5a.fit()
print(results_model_5a.summary())
#%%
#Residual Standard Error (RSE):
model_5a = sm.OLS(y, X5a).fit()
model_5a.resid.std(ddof=X5a.shape[1])
#%%
#Bland-Altman Plot:
df['BMXHIP_5'] = 50.561862 + (-0.095965*df['BMXHT']) + (0.726806*df['BMXWAIST'])
df['md_5'] = df['BMXHIP']-df['BMXHIP_5']
print(df.head(10))
#%%
df['mean_BMXHIP_5'] = statistics.mean(df['BMXHIP_5'])
df['sd_BMXHIP_5'] = statistics.stdev(df['BMXHIP_5'])
df['mean_md_5'] = statistics.mean(df['md_5'])
df['sd_md_5'] = statistics.stdev(df['md_5'])
print(df.head(10))
#%%
#Plot drawn:
print("Mean md_5:", statistics.mean(df['md_5']))
print("SD md_5: ", statistics.stdev(df['md_5']))
print("LLA: ", statistics.mean(df['md_5'])-(1.96*statistics.stdev(df['md_5'])))
print("ULA: ", statistics.mean(df['md_5'])+(1.96*statistics.stdev(df['md_5'])))

plt.plot( 'BMXHIP', 'md_5', data=df, linestyle='none', marker='o')
plt.axhline(y = 0, color = 'black', linestyle = '-')
plt.axhline(y = 1.7583862088800805e-05, color = 'blue', linestyle = '--')
plt.axhline(y = -13.99612617351395, color = 'red', linestyle = '--')
plt.axhline(y = 13.996161341238128, color = 'red', linestyle = '--')
plt.show()
#%%
#Limits of Agreement amplitude:
LOAA_5 = 13.99612617351395 + 13.996161341238128
print("LOAA_5:" , LOAA_5)
# %%
#Model 6
X6 = df[['BMXWT', 'BMXWAIST']]
# %%
model_6 = linear_model.LinearRegression()
model_6.fit(X6, y)
# %%
print(model_6.coef_)
print(model_6.intercept_)
#%%
#Add a constant:
X6a = sm.add_constant(X6)
model_6a = sm.OLS(y, X6a)
#%%
results_model_6a = model_6a.fit()
print(results_model_6a.summary())
#%%
#Residual Standard Error (RSE):
model_6a = sm.OLS(y, X6a).fit()
model_6a.resid.std(ddof=X6a.shape[1])
#%%
#Bland-Altman Plot:
df['BMXHIP_6'] = 45.073412 + (0.307425*df['BMXWT']) + (0.365595*df['BMXWAIST'])
df['md_6'] = df['BMXHIP']-df['BMXHIP_6']
print(df.head(10))
#%%
df['mean_BMXHIP_6'] = statistics.mean(df['BMXHIP_6'])
df['sd_BMXHIP_6'] = statistics.stdev(df['BMXHIP_6'])
df['mean_md_6'] = statistics.mean(df['md_6'])
df['sd_md_6'] = statistics.stdev(df['md_6'])
print(df.head(10))
#%%
#Plot drawn:
print("Mean md_6:", statistics.mean(df['md_6']))
print("SD md_6: ", statistics.stdev(df['md_6']))
print("LLA: ", statistics.mean(df['md_6'])-(1.96*statistics.stdev(df['md_6'])))
print("ULA: ", statistics.mean(df['md_6'])+(1.96*statistics.stdev(df['md_6'])))

plt.plot( 'BMXHIP', 'md_6', data=df, linestyle='none', marker='o')
plt.axhline(y = 0, color = 'black', linestyle = '-')
plt.axhline(y = -2.641691317246042e-06, color = 'blue', linestyle = '--')
plt.axhline(y = -12.79523587443649, color = 'red', linestyle = '--')
plt.axhline(y = 12.795230591053855, color = 'red', linestyle = '--')
plt.show()
#%%
#Limits of Agreement amplitude:
LOAA_6 = 12.79523587443649 + 12.795230591053855
print("LOAA_6:" , LOAA_6)
# %%
#Model 7
X7 = df[['BMXHT','BMXWT', 'BMXWAIST']]
# %%
model_7 = linear_model.LinearRegression()
model_7.fit(X7, y)
# %%
print(model_7.coef_)
print(model_7.intercept_)
#%%
#Add a constant:
X7a = sm.add_constant(X7)
model_7a = sm.OLS(y, X7a)
#%%
results_model_7a = model_7a.fit()
print(results_model_7a.summary())
# %%
#Residual Standard Error
model_7a = sm.OLS(y, X7a).fit()
model_7a.resid.std(ddof=X7a.shape[1])
# %%
#Bland_Altman Plot:
df['BMXHIP_7'] = 130.824112 + (-0.488518*df['BMXHT']) + (0.617183*df['BMXWT']) + (0.064531*df['BMXWAIST'])
df['md_7'] = df['BMXHIP']-df['BMXHIP_7']
print(df.head(10))
#%%
df['mean_BMXHIP_7'] = statistics.mean(df['BMXHIP_7'])
df['sd_BMXHIP_7'] = statistics.stdev(df['BMXHIP_7'])
df['mean_md_7'] = statistics.mean(df['md_7'])
df['sd_md_7'] = statistics.stdev(df['md_7'])
print(df.head(10))
#%%
#Plot drawn:
print("Mean md_7:", statistics.mean(df['md_7']))
print("SD md_7: ", statistics.stdev(df['md_7']))
print("LLA: ", statistics.mean(df['md_7'])-(1.96*statistics.stdev(df['md_7'])))
print("ULA: ", statistics.mean(df['md_7'])+(1.96*statistics.stdev(df['md_7'])))

plt.plot( 'BMXHIP', 'md_7', data=df, linestyle='none', marker='o')
plt.axhline(y = 0, color = 'black', linestyle = '-')
plt.axhline(y = -4.144073241301843e-05, color = 'blue', linestyle = '--')
plt.axhline(y = -10.497097910321404, color = 'red', linestyle = '--')
plt.axhline(y = 10.497180791786231, color = 'red', linestyle = '--')
plt.show()
#%%
#Limits of Agreement amplitude:
LOAA_7 = 10.497097910321404 + 10.497180791786231
print("LOAA_7:" , LOAA_7)
# %%
#CC between models and BMXHIP:
print(pearsonr(df["BMXHIP"], df["BMXHIP_1"]))
print(pearsonr(df["BMXHIP"], df["BMXHIP_2"]))
print(pearsonr(df["BMXHIP"], df["BMXHIP_3"]))
print(pearsonr(df["BMXHIP"], df["BMXHIP_4"]))
print(pearsonr(df["BMXHIP"], df["BMXHIP_5"]))
print(pearsonr(df["BMXHIP"], df["BMXHIP_6"]))
print(pearsonr(df["BMXHIP"], df["BMXHIP_7"]))
# %%
#Z-Score:
import numpy as np
#%%
from numpy import linspace
#%%
import pandas as pd
#%%
import seaborn as sns
#%%
import matplotlib.pyplot as plt
#%%
from scipy.stats import gaussian_kde
#%%
df["mean_BMXHIP"] = statistics.mean(df["BMXHIP"])
print("Mean BMXHIP: ", df["mean_BMXHIP"])
#%%
df["sd_BMXHIP"] = statistics.stdev(df["BMXHIP"])
print("SD BMXHIP: ", df["sd_BMXHIP"])
#%%
df["zscore_BMXHIP"] = ((df["BMXHIP"])-(df["mean_BMXHIP"]))/(df["sd_BMXHIP"])
print("Z-score BMXHIP: ", df["zscore_BMXHIP"])
#%%
df["mean_zscore_BMXHIP"] = statistics.mean(df["zscore_BMXHIP"])
print("Mean z-score BMXHIP: ", df["mean_zscore_BMXHIP"])
# %%
#Model 1:
df["zscore_BMXHIP_1"] = ((df["BMXHIP_1"])-(df["mean_BMXHIP"]))/(df["sd_BMXHIP"])
df["mean_zscore_BMXHIP_1"] = statistics.mean(df["zscore_BMXHIP_1"])
print("Mean Z-score BMXHIP 1: ", df["mean_zscore_BMXHIP_1"])
#%%
#Graph:
sns.kdeplot(df["zscore_BMXHIP"], shade=True, color="orange")
kde = gaussian_kde(df["zscore_BMXHIP_1"])
x_range = linspace(min(df["zscore_BMXHIP_1"]), max(df["zscore_BMXHIP_1"]), len(df["zscore_BMXHIP_1"]))
sns.lineplot(x=x_range*-1, y=kde(x_range) * -1, color='lightgreen') 
plt.fill_between(x_range*-1, kde(x_range) * -1, color='lightgreen')
plt.xlabel("Z-Score")
plt.axhline(y=0, linestyle='-',linewidth=1, color='black')
plt.show()
# %%
#Model 2:
df["zscore_BMXHIP_2"] = ((df["BMXHIP_2"])-(df["mean_BMXHIP"]))/(df["sd_BMXHIP"])
df["mean_zscore_BMXHIP_2"] = statistics.mean(df["zscore_BMXHIP_2"])
print("Mean Z-score BMXHIP 2: ", df["mean_zscore_BMXHIP_2"])
#%%
#Graph:
sns.kdeplot(df["zscore_BMXHIP"], shade=True, color="orange")
kde = gaussian_kde(df["zscore_BMXHIP_2"])
x_range = linspace(min(df["zscore_BMXHIP_2"]), max(df["zscore_BMXHIP_2"]), len(df["zscore_BMXHIP_2"]))
sns.lineplot(x=x_range*-1, y=kde(x_range) * -1, color='lightgreen') 
plt.fill_between(x_range*-1, kde(x_range) * -1, color='lightgreen')
plt.xlabel("Z-Score")
plt.axhline(y=0, linestyle='-',linewidth=1, color='black')
plt.show()
# %%
