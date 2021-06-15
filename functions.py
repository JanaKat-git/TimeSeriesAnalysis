'''
Script with Functions to
- clean Timeseries Data from ECA&D and save the cleaned Data as csv-file
- analyze Trend, Seasonaliyt and Remainder
- calculate the # of lags for an AutoReg Model
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.linear_model import LinearRegression

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


def explore_dataset(csv_file, location):
    '''
    Vizualisation of the Temperature dataset(csv-file) 

    Parameters
    ----------
    csv_file: str
        Name of the csv-file
    location: str
        The location of the weatherstation

    Returns
    ---------
    
    '''
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    #check for null values
    plt.bar(x='Nan', height=df.isna().sum())
    plt.title('Check for NaN values')
    plt.show()

    plt.plot(df.index,df['t_mean'])
    plt.xlabel('Year')
    plt.ylabel('Avg. Temperature [°C]')
    plt.title('Temperature profile: ' + location)
    plt.show()
    plt.clf()

    plt.plot(df.index[:3650],df['t_mean'][:3650])
    plt.xlabel('dates')
    plt.ylabel('Avg. Temperature [°C]')
    plt.title('Temperature profile at '+location+ ' the last 10 years.')
    plt.show()
    plt.clf()

    plt.plot(range(1,13), df.groupby(df.index.month)['t_mean'].mean())
    plt.xticks(rotation=30)
    plt.xlabel('Months')
    plt.ylabel('Avg. Temperature for each Month in the year range [°C]')
    plt.title('Mean Temperature profile per Month: ' + location)
    plt.show()
    plt.clf()


def load_split_data_month(csv_file):
    '''
    Load of the Temperature dataset(csv-file) and split into train and test data. 
    Creating the index column with monthly frequency and the monthly mean of the data.

    Parameters
    ----------
    csv_file: str
        Name of the csv-file
    
    Returns
    ---------
    df_m: pd.dataframe()
        The dataframe with the train-data with an monthly freq.
    df_test: pd.dataframe()
        The dataframe with the test-data with an monthly freq.
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
    df_test = df_data[-356::] #create a test DataFrame df with values of last year
    df = df_data[:(len(df_data)-356)] #create a train DataFrame with all values expect one year
    
    df_m = df.resample('MS').mean() #instead of daily mean use monthly mean
    df_m.dropna(axis=0,inplace = True)
    df_test = df_test.resample('MS').mean()

    Xtrain = df_m.index 
    ytrain = df_m['t_mean']

    Xtest = df_test.index
    ytest = df_test['t_mean']

    return df_m, df_test, Xtrain, ytrain, Xtest, ytest


def analyzing_trend_month(df,y):
    '''
    Analyzing the Trend of the Temperature dataset by creating a timestep column 
    and modeling a Linear Regression with and without Polynomial Features.

    Parameters
    ----------
    df: pd.dataframe()
        The dataframe with the train-data
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

    trend = m.coef_*12*df.index.year.nunique()
    print('Trend: '+str(trend)+' °C') #getting a trend 
    print('intercept: ' + str(m.intercept_)) #getting an intercept
    
    df['trend'] = m.predict(Xtrend) 
    df[['t_mean', 'trend']].plot()
    plt.xlabel('Year')
    plt.ylabel('Avg. Temperature [°C]')
    plt.title('Trend Linear')
    plt.show()
   

    #Modelling polynomial 
    column_poly = ColumnTransformer([('poly', PolynomialFeatures(degree=2, include_bias=False), ['timestep'])]) #using polyFeat to analyze polynomial behaviour of Temp over time
    
    column_poly.fit(Xtrend) 
    Xtrans=column_poly.transform(Xtrend) 
    
    m_poly = LinearRegression()
    m_poly.fit(Xtrans, y)

    trend = m_poly.coef_*12*df.index.year.nunique() 
    print('trend: ' + str(trend))

    df['trend_poly'] = m_poly.predict(Xtrans)
    df[['t_mean','trend_poly']].plot()
    plt.xlabel('Year')
    plt.ylabel('Avg. Temperature [°C]')
    plt.title('Trend Polynomial Degree = 2')
    plt.show()
   
    return df


def analyzing_seasonality_month(df,X,y):
    '''
    Analyzing the Seasonality of the Temperature dataset using the month column.

    Parameters
    ----------
    df: pd.dataframe()
        The whole DataFrame
    Xtrain: matrix
        The X values to train the model
    ytrain: vector
        The y values to train the model
    
    Returns
    ---------
    df: pd.dataframe()
        The new dataframe with the trendANDseasonality column and the dummies(month).
    '''
    
    df['month'] = df.index.month #create column with # of month

    seasonal_dummies = pd.get_dummies(df['month'],prefix='month', drop_first=True) #create dummies for each month

    df = df.merge(seasonal_dummies,left_index = True, right_index=True)
    Xseason = df.drop(['t_mean','trend','trend_poly','month'], axis=1)
    
    m = LinearRegression()
    m.fit(Xseason, y)

    df['trendANDseasonality'] = m.predict(Xseason)
      
    df[['t_mean','trendANDseasonality']].plot()
    plt.xlabel('Year')
    plt.ylabel('Avg. Temperature [°C]')
    plt.title('Trend and Seasonality')
    plt.show()

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
        The dataframe with all columns.
    csv-file in the folder
    '''
    # Remainder
    df['remainder'] = df['t_mean'] - df['trendANDseasonality'] #calculate the remainder by subtracting trend ans seasonality

    df['remainder'].plot() 
    plt.xlabel('Year')
    plt.ylabel('Avg. Temperature [°C]')
    plt.title('Remainder')
    plt.show()

    df_re = df['remainder']
    df_re.to_csv(csv_output_name)

    return df


def number_lags(csv_file):
    '''
    Using statsmodel to find out how many lags should be used for the AutoReg Model.

    Parameters
    ----------
    csv_file: str
        The Name of the csv-file with the data
    
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
    selected_order = ar_select_order(endog=df_re['remainder'], maxlag=12, old_names=False)
    lags = selected_order.ar_lags #show the selected lags

    m_ar = AutoReg(endog=df_re['remainder'], lags=3).fit() #use statsmodel and integrated autoreg model
    summary = m_ar.summary()
    
    return df_re, lags, summary



   

    


    



