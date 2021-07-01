# Time Series Analysis

## Description
Time Series Analysis for short-term temperature forecasting with the creation of a basic model for the modelling of trend and seasonality and the modelling of the
Modelling the time dependence of the residual using an AR mode.

``plot_temp_month.ipynb``: Notebook with temperature curve at the weather station 'Zugspitze' for each month from 1945 - 2020.

``forcast.py``: Prediction of the average temperature of a month by using TimeSeries Data and analysing the
Trend, Seasonality and Remainder of the Data. Using an AutoReg Model for the Prediction.

``functions.py``
Functions for:
- cleaning Timeseries Data from ECA&D and save the cleaned Data as csv-file
- analysing Trend, Seasonality and Remainder
- calculating the # of lags for an AutoReg Model

