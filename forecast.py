'''
Prediction of the average Temperture of a Month by using TimeSeries Data and analyzing the
Trend, Seasonality and Remainder of the Data. Using an AutoReg Model for the Prediction.
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
import functions as f


df = f.data_cleaning('./data/DATA.txt')

f.explore_dataset('./data/DATA_CLEAN.csv', 'location')


#Train and Test split
df_m, df_test, Xtrain, ytrain, Xtest, ytest= f.load_split_data_month('./data/DATA_CLEAN.csv')

#Calculating the trend
df_trend = f.analyzing_trend_month(df_m,ytrain)

#Calculating Seasonality and predict trendANDseasonality
df_season_and_trend = f.analyzing_seasonality_month(df_trend, Xtrain, ytrain)

#Calculating the remaninder and saving a csv file with dates and remainder
df_re = f.analyzing_remainder(df_season_and_trend, './data/remainder_zs_month.csv')

#Analyzing number of lags for AutoReg Model
df_re, lags, summary = f.number_lags('./data/remainder_zs_month.csv')

#Add lags
df_re['lag1'] = df_re['remainder'].shift(1) 
#df_re['lag2'] = df_re['remainder'].shift(2) 
#df_re['lag3'] = df_re['remainder'].shift(3) 
df_re.dropna(inplace=True) #drop NaN values created during creating the lags

#Heatmap
corre = round(df_re.corr(),2) #correlation between lag and remainder
sns.heatmap(corre, annot=True) #create heatmap
plt.show()

#create scatterplot of lags and remainder
plt.clf()
plt.scatter(df_re['lag1'], df_re['remainder'])
#plt.scatter(df_re['lag2'], df_re['remainder'])
#plt.scatter(df_re['lag3'], df_re['remainder'])
plt.title('remainder vs. lag')
plt.ylabel('remainder')
plt.xlabel('lag values')


Xar = df_re[['lag1',
#'lag2',
#'lag3'
]] 
yar = df_re[['remainder']] 

m = LinearRegression()    
m.fit(Xar, yar) 

df_re['pred_remainder'] = m.predict(Xar) 

#create plot with lag1
df_re[['remainder','lag1','pred_remainder']].plot()
plt.title('remainder and prediction of remainder with one lag')
plt.ylabel('Avg. Temperature [°C]')
plt.xlabel('Year')
plt.show()

#calculate residual
df_re['residual'] = df_re['remainder'] - df_re['pred_remainder']

#plot the residual
plt.plot(df_re.index, df_re['residual'])
plt.title('residual of predicted remainder with one lag')
plt.ylabel('Avg. Temperature [°C]')
plt.xlabel('Year')
plt.show()


#Prediction with AutoReg model
df_full = df_season_and_trend.merge(df_re,left_index = True, right_index=True, how='inner')
yfull = df_full['t_mean']


m_ar = AutoReg(yfull, lags=1).fit()
m_ar.predict(start='2020-04-01', end='2020-04-01')