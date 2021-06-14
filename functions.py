'''
Script to clean Timeseries Data from ECA&D and save the cleaned Data as csv-file.
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

def data_cleaning(datapath):
    '''
    Function to clean Timeseries Data from ECA&D and save the cleaned Data as csv-file.

    Parameters
    --------
    datapath: str
        File path to the data.

    Return
    df: pd.DataFrame    
        Cleaned DataFrame

    '''

    df = pd.read_table(datapath, skiprows = 18, delimiter= ',', header=0, index_col=1, parse_dates = True)

    df.rename(columns = {'   TG':'TG', ' SOUID': 'SOUID', ' Q_TG': 'Q_TG'}, inplace = True) #remove whitespaces in header, rename columns
    df.index.rename('date', inplace = True) #rename index column
    df.dropna(axis=0,inplace = True) #drops zero values
    df.index = pd.to_datetime(df.index) #transform in pandas DataFrame

    df = df.replace('\n','', regex=True) #replace newlines
    df = df.replace(' ','', regex=True) #replace whitespaces

    df['t_mean'] = df['TG']*0.1 #TG is given in 0.1°C create column with T in °C
    df['t_mean'].round(decimals = 2) 
    df.drop(df[df['t_mean'] < -30].index, inplace = True) #drop values on index where T is smaller than -30 °C (false data)
    df.drop(df[['SOUID', 'Q_TG', 'TG']], axis =1, inplace = True) #drop SOUID, Q_TG and TG column

    df.to_csv('./data/DATA_CLEAN.csv') #save cleaned DataFrame

    return df


def explore(csv_file, location):
    '''
    Vizualisation of the Temperature dataset(csv-file) 

    Parameters
    ----------
    csv_file: str
        Name of the Dataset
    location: str
        The location of the weatherstation

    Returns
    ---------
    
    '''
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    #check for null values
    plt.bar(x='Nan', height=df.isna().sum())
    plt.show()

    #Plot the data
    dates = df.index #assign x
    temp = df['t_mean']#assign y

    plt.clf()
    plt.plot(dates,temp)
    plt.xlabel('dates')
    plt.ylabel('Mean Temperature')
    plt.title('Temperature profile: ' + location)
    plt.show()

    dates_2020 = df.index[-3650:]
    temp_2020 = df['t_mean'][-3650:]
    plt.plot(dates_2020,temp_2020)
    plt.xlabel('dates')
    plt.ylabel('Mean Temperature')
    plt.title('Temperature profile at '+location+ ' the last 10 years.')
    plt.show()

    df['month'] = df.index.month #create column with no. of month
  
    plt.plot(range(1,13), df.groupby(df.index.month)['t_mean'].mean()) #plot the mean T of each month against the months
    plt.xlabel('Months')
    plt.ylabel('Mean Temperature for each month in the year range')
    plt.title('Mean Temperature profile per Month: ' + location)
    plt.show()
    return 


def load_split_data_month(csv_file):
    '''
    Load of the Temperature dataset(csv-file) and split into train and test data. 
    Creating the index column with monthly frequ. and the monthly mean of the data.

    Parameters
    ----------
    csv_file: str
        The data to load
    
    Returns
    ---------
    df_m: pd.dataframe()
        The dataframe with the traindata with an monthly freq.
    df_test: pd.dataframe()
        The dataframe with the testdata with an monthly freq.
    Xtrain: matrix
        The X values to train the model
    ytrain: vector
        The y values to train the model
    Xtest: matrix
        The X values to test the model
    ytest: vector
        The y values to test the model
    '''
    #load data
    df_data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    #Train_test-split
    df_test = df_data[-356::] #create test df with values of last year
    df = df_data[:(len(df_data)-356)] #create train df with all values expect one year
    
    df_m = df.resample('MS').mean() #instead of daily mean use monthly mean
    df_test = df_test.resample('MS').mean()

    Xtrain = df_m.index 
    ytrain = df_m['t_mean']

    Xtest = df_test.index
    ytest = df_test['t_mean']

    return df_m, df_test, Xtrain, ytrain, Xtest, ytest



def analyzing_trend(df,y):
    '''
    Analyzing the Trend of the Temperature dataset by creating a timestep column and modeling a LinReg with and without PolyFeatures.

    Parameters
    ----------
    df: pd.dataframe()
        The dataframe with the traindata
    ytrain: vector
        The y values to train the model
    
    Returns
    ---------
    df: pd.dataframe()
        The new dataframe with the timestep, trend and trend_poly column.
    '''

   # Analyzing Time Series
    df['timestep'] = range(len(df)) #creating timesteps to get the trend out of the data with a LinearRegression

    Xtrend = df[['timestep']] #assign X values for LinReg

    #Modelling linear
    m = LinearRegression()
    m.fit(Xtrend, y)

    trend = m.coef_*12*74 #calculation to get difference in Temperature over the given time: coeff_*12months*74years
    print('Trend: '+str(trend)+' °C') #getting a trend 
    print('intercept: ' + str(m.intercept_)) #getting an intercept
    
    df['trend'] = m.predict(Xtrend) #create trend column in df_t
    df[['t_mean', 'trend']].plot()#plot the trend


    #Modelling polynomial 
    column_poly = ColumnTransformer([('poly', PolynomialFeatures(degree=2, include_bias=False), ['timestep'])]) #using polyFeat to analyze polynomial behaviour of Temp over time
    
    column_poly.fit(Xtrend) #fit data
    Xtrans=column_poly.transform(Xtrend) #transform xdata
    
    #LineRef
    m_poly = LinearRegression()
    m_poly.fit(Xtrans, y)

    trend = m_poly.coef_*12*74 #calculation to get difference in Temperature over the given time: coeff_*12months*74years
    print('trend: ' + str(trend))

    df['trend_poly'] = m_poly.predict(Xtrans) #create poly_trend column
    df[['t_mean','trend_poly']].plot() #plot the trend_poly result
    return df


def analyzing_trend_monthly(df):
    '''
    Analyzing the change of the Meantemperature over a given timerange.
    
    Parameters
    ----------
    df: pd.dataframe()
        The dataframe with the temperaturedata
    
    Returns
    ---------
    trend: int
        The Temperature change of a given month.
    '''

   # Analyzing Time Series
    df['timestep'] = range(len(df)) #creating timesteps to get the trend out of the data with a LinearRegression

    Xtrend = df[['timestep']] #assign X values for LinReg
    y = df['t_mean']
    #Modelling linear
    m = LinearRegression()
    m.fit(Xtrend, y)

    trend = m.coef_*74 #calculation to get difference in Temperature over the given time: coeff_*12months*74years
    
    return trend


def analyzing_seasonality(df,X,y):
    '''
    Analyzing the Seasonality of the Temperature dataset with the month column.

    Parameters
    ----------
    df: pd.dataframe()
        The dataframe with the data
    Xtrain: matrix
        The X values to train the model
    ytrain: vector
        The y values to train the model
    
    Returns
    ---------
    df: pd.dataframe()
        The new dataframe with the trendANDseasonality column and the dummies(month).
    '''
    
    df['month'] = df.index.month #create column with no. of month

    seasonal_dummies = pd.get_dummies(df['month'],prefix='month', drop_first=True) # create dummies for each month

    df = df.merge(seasonal_dummies,left_index = True, right_index=True) #merge df seasonal_dummies and df

    Xseason = df.drop(['t_mean','trend','trend_poly','month'], axis=1) #choose X values for LinReg fit (all dummy columns)
    
    #LinReg
    m = LinearRegression()
    m.fit(Xseason, y)

    df['trendANDseasonality'] = m.predict(Xseason) #create trend and seasonality column in df
      
    df[['t_mean','trendANDseasonality']].plot()#plot the seasonality and trend
    return df



def analyzing_seasonality_daily(df,X,y):
    '''
    Analyzing the Seasonality of the Temperature dataset with the day column.

    Parameters
    ----------
    df: pd.dataframe()
        The dataframe with the data
    Xtrain: matrix
        The X values to train the model
    ytrain: vector
        The y values to train the model
    
    Returns
    ---------
    df: pd.dataframe()
        The new dataframe with the trendANDseasonality column and the dummies(356 days).
    '''
    df = df
    
    df['day'] = df.index.dayofyear #create column with no. of month
    
    seasonal_dummies_day = pd.get_dummies(df['day'],prefix='day_') # create dummies for each month

    df = df.merge(seasonal_dummies_day,left_index = True, right_index=True) #merge dummies df and df
    
    Xseason = df.drop(['t_mean','trend','trend_poly','day'], axis=1) #choose X values for LinReg fit (all dummy columns)
    m = LinearRegression()
    m.fit(Xseason, y)

    df['trendANDseasonality'] = m.predict(Xseason) #create trend and seasonality column in df
      
    df[['t_mean','trendANDseasonality']].plot()#plot the seasonality and trend
    return df


    
def analyzing_remainder(df, csv_output_name):
    '''
    Analyzing the Remainder and save th df in a csv file.

    Parameters
    ----------
    df: pd.dataframe()
        The dataframe with the data
    csv_output_name: str
        The filename for the output data.
    
    Returns
    ---------
    df: pd.dataframe()
        The dataframe with all columns: .
    csv-file in the folder
    '''
    # Remainder
    df['remainder'] = df['t_mean'] - df['trendANDseasonality'] #calculate the remainder by subtracting trend ans seasonality
    df['remainder'].plot() # plot the remainder
    df_re = df['remainder']
    df_re.to_csv(csv_output_name)#save the df as csv file
    return df



def autoregressive_model_remainder(csv_file):
    '''
    Using an autoregressive model to predict the rmainder of the data. Using the LinReg and the AutoReg from statsmodels

    Parameters
    ----------
    csvfile: str
        The csvfile with the data
    
    Returns 
    ---------
    df_re: pd.Dataframe
        The dataframe of the csv_file with the lag columns, prediction and residuals
    lags: lst
        The amount of lags that was calculated with ar_select_order (statsmodel)
    '''
    #load remainder csv file
    df_re = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    #Add lag
    df_re['lag1'] = df_re['remainder'].shift(1) #add lag1 to df
    df_re['lag2'] = df_re['remainder'].shift(2) #add lag2 to df
    df_re['lag3'] = df_re['remainder'].shift(3) #add lag3 to df
    
    df_re.dropna(inplace=True) #drop NaN values created during creating the lags

    corre = round(df_re.corr(),2) #correlation between lag and remainder
    sns.heatmap(corre, annot=True) #create heatmap
    plt.show()

    #create scatterplot of lag1 and remainder
    plt.clf()
    plt.scatter(df_re['lag1'], df_re['remainder'])
    #plt.scatter(df_re['lag2'], df_re['remainder'])
    #plt.scatter(df_re['lag3'], df_re['remainder'])
    plt.title('remainder vs. lag')
    plt.ylabel('remainder')
    plt.xlabel('lag')


    Xar = df_re[['lag1',
    'lag2',
    'lag3'
    ]] # assign X for LinReg
    yar = df_re[['remainder']] #assign y for Lin Reg

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

    #Plot Partial Autocorrelation to find out how many lags sholud be used
    plot_pacf(df_re['remainder'])

    #Use function from statsmodel to find out how many lags should be used
    selected_order = ar_select_order(endog=df_re['remainder'], maxlag=12, old_names=False) #Let ar_select_order select the number of lags for the remainder
    lags = selected_order.ar_lags #show the selected lags

    m_ar = AutoReg(endog=df_re['remainder'], lags=3).fit() #use statsmodel and integrated autoreg model
    print(m_ar.summary())
    return df_re, lags


def number_lags(csv_file):
    '''
    Using statsmodel to find out how many lags should be used for the ML.

    Parameters
    ----------
    csvfile: str
        The csvfile with the data
    
    Returns 
    ---------
    df_re: pd.Dataframe
        The dataframe of the csv_file.
    lags: lst
        The amount of lags that was calculated with ar_select_order (statsmodel)
    summary: stasmodel output after fitting the model 
    '''
    #load remainder csv file
    df_re = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    #Plot Partial Autocorrelation and Autocorrlation to find out which model (AR, MA or ARIMA to use)
    plot_pacf(df_re['remainder'])
    plot_acf(df_re['remainder'])
    
    #Use function from statsmodel to find out how many lags should be used
    selected_order = ar_select_order(endog=df_re['remainder'], maxlag=12, old_names=False) #Let ar_select_order select the number of lags for the remainder
    lags = selected_order.ar_lags #show the selected lags

    m_ar = AutoReg(endog=df_re['remainder'], lags=3).fit() #use statsmodel and integrated autoreg model
    summary = m_ar.summary()
    
    return df_re, lags, summary



def full_model(df_trend_seasonality, df_lags):
    '''
    Using LinReg model to predict the daily Temp in future.

    Parameters
    ----------
    df_trend_seasonality: pd.Dataframe
        The df with the trend and seasonality data
    df_lags: pd.Dataframe
        The df with the lags.
    
    Returns 
    ---------
    df_full: pd.Dataframe
        The full dataframe with all columns for prediction.
    m_full: sklearn.LinearRegaression()
        model to fit the data

    '''
    df_full = df_trend_seasonality.merge(df_lags,left_index = True, right_index=True, how='inner')

    Xfull = df_full.drop(columns=['t_mean','trend', 'trend_poly','month','remainder','residual','pred_remainder'])
    yfull = df_full['t_mean']

    m_full = LinearRegression()
    m_full.fit(Xfull, yfull)

    df_full['pred_full'] = m_full.predict(Xfull)

    return m_full, df_full



def cross_val(X,y,m):
    '''
    Cross Validation of the modeled data.

    Parameters
    ----------
    X: Matrix 
        The X Matrix with all the data of the df for the cross validation.
    y: vector
        The y vector with the data of the df for the cross validation
    m: sklearn.LinearRegaression()
        model which was used to fit the data
    
    Returns 
    ---------
    result: list
        Result of the cross validation
    '''
    ts_split = TimeSeriesSplit(n_splits = 10)
    ts_split.split(X, y)
    time_series_split = ts_split.split(X, y) 
    result = cross_val_score(estimator=m, X=X, y=y, cv=time_series_split)
    return result



def feature_eng(df):
    '''
    Feature Engineering (FE) of the pd.Dataframe to use in a ML-model to predict Temp. data in the future.

    Parameters
    ----------
    df: pd.Datafram 
        The df for the FeatureEngineering.
    
    Returns 
    ---------
    df_trans: pd.Datafram
        The transformed df with all FE columns
    '''
    df['timestep'] = range(len(df))
    df['month'] = df.index.month #create column with no. of month

    seasonal_dummies = pd.get_dummies(df['month'],prefix='month', drop_first=True) # create dummies for each month

    df_trans = df.merge(seasonal_dummies,left_index = True, right_index=True)

    X = df_trans.drop(['t_mean','month'], axis=1)
    y = df_trans['t_mean']

    column_poly = ColumnTransformer([('poly', PolynomialFeatures(degree=2, include_bias=False), ['timestep'])]) #using polyFeat to analyze polynomial behaviour of Temp over time
    column_poly.fit(X) #fit data

    Xtrans=column_poly.transform(X) #transform xdata
    
    #LineReg
    m_poly = LinearRegression()
    m_poly.fit(Xtrans, y)
    
    df_trans['trendANDseason'] = m_poly.predict(Xtrans) #create poly_trend column

    df_trans['remainder'] = df_trans['t_mean'] - df_trans['trendANDseason'] #calculate the remainder by subtracting trend ans seasonality
    
    df_trans['lag1'] = df_trans['remainder'].shift(1) #add lag1 to df
    df_trans['lag2'] = df_trans['remainder'].shift(2) #add lag2 to df
    df_trans['lag3'] = df_trans['remainder'].shift(3) #add lag3 to df
    df_trans.dropna(inplace=True)

    return df_trans



   

    


    



