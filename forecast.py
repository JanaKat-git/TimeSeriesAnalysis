
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import functions as f

df = f.data_cleaning('./data/DATA')


#load data for model
df_m, df_test, Xtrain, ytrain, Xtest, ytest= f.load_split_data_month('./data/DATA_CLEAN.csv')

#Calculating  the trend
df_trend = f.analyzing_trend(df_m, ytrain)

#Calculating Seasonality and predict trendANDseasonality
df_season_and_trend = f.analyzing_seasonality(df_trend, Xtrain, ytrain)

df_season_and_trend[['t_mean','trendANDseasonality']].loc['1970-01-01':'1980-12-01'].plot()

#Calculating the remaninder and saving a csv file with dates and remainder
df_remainder = f.analyzing_remainder(df_season_and_trend, 'remainder_zs_month.csv')

#Calculating number of lags and Model for Data
df_re, lags, summary = f.number_lags('remainder_zs_month.csv')
print(lags)


#create df with lags
df_re = pd.read_csv('remainder_zs.csv', index_col=0, parse_dates=True)

#Add lags
df_re['lag1'] = df_re['remainder'].shift(1) #add lag1 to df
df_re.dropna(inplace=True)

corre = round(df_re.corr(),2) #correlation between lag and remainder
sns.heatmap(corre, annot=True) #create heatmap
plt.show()

# assign X for LinReg
Xar = df_re[['lag1',
]] 

#assign y for Lin Reg
yar = df_re[['remainder']] 

#Modelling with LinReg
m = LinearRegression() #model   
m.fit(Xar, yar) #fit values to model

df_re['pred_remainder'] = m.predict(Xar) #predict values with model

#create plot with lag1
df_re[['remainder','lag1','pred_remainder']].plot()
plt.show()

#calculate residual
df_re['residual'] = df_re['remainder'] - df_re['pred_remainder']

#plot the residual
plt.plot(df_re.index, df_re['residual'])
plt.title('residual')
plt.show()

#create df with trend and seasonality
df_season_and_trend = F.analyzing_seasonality(df_m, Xtrain, ytrain)


#Modelling th full data(trend, seasonality, remainder)
df_full = df_season_and_trend.merge(df_re,left_index = True, right_index=True, how='inner')

Xfull = df_full.drop(columns=['t_mean','trend', 'trend_poly','month','remainder','residual','pred_remainder','trendANDseasonality'])
yfull = df_full['t_mean']

m_full = LinearRegression()
m_full.fit(Xfull, yfull)

df_full['pred_full'] = m_full.predict(Xfull)

m_full.score(Xfull,yfull)

#CrossValdiation of the full model
result = F.cross_val(Xfull,yfull,m_full)

print(result.mean())


#Using test data
df_trans_test = F.feature_eng(df_test)
df_trans_test.head()

Xtest = df_trans_test.drop(columns=[
    't_mean',
    'month', 'remainder','trendANDseason','lag2','lag3'
    ])
ytest = df_trans_test['t_mean']

m_full.score(Xtest,ytest)


#Prediction with AR model
m_ar = AutoReg(yfull, lags=1).fit()
m_ar.predict(start='2020-05-01', end='2020-05-01')