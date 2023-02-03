import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from skforecast.ForecasterAutoreg import ForecasterAutoreg

from sklearn.linear_model import Lasso


from sklearn.metrics import mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_file_search(file, period, steps, search, regressor, lags, plot = True):
    '''
    Read the data and forecast for chosen search term with chosen regressor
    adapted from https://joaquinamatrodrigo.github.io/skforecast/0.3/guides/autoregresive-forecaster-exogenous.html 
    Input:
    file - lowercase name of the file 
    period - last date for the data, string, format Y-m-d, maxinmum 2023-01-01
    steps - int, test split (number of weeks for test data and predicting)
    search - string, search term
    regressor - name of regressor for forecasting model 
    lags - int, on what time perion to forecast
    plot - boolean, do we want a plot
    
    Output:
    State and Search term
    MSE 
    Predictions vs actual plot
        
    Return:
    MSE
    '''
    # Print state and search
    print(f'\n Forecast for {file} and {search} untill {period} \n' )
    
    # Read the data
    df = pd.read_csv(f'../data/{file}.csv')
    df['week'] = pd.to_datetime(df['week'], format = '%Y-%m-%d')
    df = df[df['week'] < period]
    df.set_index('week', inplace=True)
    
    # Change to frequency
    df = df.asfreq('W')
    
    # Split the data to train-test
    steps = steps
    data_train = df.iloc[:-steps, :]
    data_test  = df.iloc[-steps:, :]
    
    # Create and fit forecaster
    forecaster = ForecasterAutoreg(
                    regressor = regressor,
                    lags      = lags
                )

    forecaster.fit(
        y    = data_train[search],
        exog = data_train[['stay_at_home', 'mask_mandate', 'gatherings_banned', 'business_closures', 'travel_restrictions']]
    )
      
    # predict
    predictions = forecaster.predict(
                steps = steps,
                exog = data_test[['stay_at_home', 'mask_mandate', 'gatherings_banned', 'business_closures', 'travel_restrictions']]
               )
    
    # Add datetime index to predictions
    predictions = pd.Series(data=predictions, index=data_test.index)
    
    # Calculate MSE
    error_mse = round(mean_squared_error(
                y_true = data_test[search],
                y_pred = predictions
            ),2)
    
    
    print(f"Test error (mse): {error_mse} \n")

    if plot == True:
        # Plot predictions vs actual
        fig, ax=plt.subplots(figsize=(11, 6))
        data_train[search].plot(ax=ax, label='train')
        data_test[search].plot(ax=ax, label='test')
        predictions.plot(ax=ax, label='predictions')
        ax.legend()
        ax.set_title(f'Forecast for {file} and {search}', size = 20)
        ax.set_ylabel(f'{search}', size = 20)
        ax.set_xlabel('Week', size = 20)
        plt.plot();

    return error_mse
    
        
        
        
        
        
def forecast_file_search_without_exogin(file, period, steps, search, regressor, lags):
    '''
    Read the data and forecast for chosen search term with chosen regressor
    adapted from https://joaquinamatrodrigo.github.io/skforecast/0.3/guides/autoregresive-forecaster-exogenous.html 
    Input:
    file - lowercase name of the file 
    period - last date for the data, string, format Y-m-d, maxinmum 2023-01-01
    steps - int, test split (number of weeks for test data and predicting)
    regressor - name of regressor for forecasting model 
    lags - int, on what time perion to forecast
    
    Output:
    State and Search term
    Forecaster output
    First 3 predictions
    Predictions vs actual plot
    MSE 
    Feature importances
    
    Return:
    Forecaster
    '''
    # Print state and search
    print(f'\n Forecast for {file} and {search} untill {period} \n' )
    
    # Read the data for the state
    df = pd.read_csv(f'../data/{file}.csv')
    df['week'] = pd.to_datetime(df['week'], format = '%Y-%m-%d')
    df = df[df['week'] < period]
    df.set_index('week', inplace=True)
    
    # Change to frequency
    df = df.asfreq('W')
    
    # Split the data to train-test
    steps = steps
    data_train = df.iloc[:-steps, :]
    data_test  = df.iloc[-steps:, :]
    
    # Create and fit forecaster
    forecaster = ForecasterAutoreg(
                    regressor = regressor,
                    lags      = lags
                )

    forecaster.fit(
        y    = data_train[search],
    )
        
    # predict
    predictions = forecaster.predict(
                steps = steps,
               )
    
    # Add datetime index to predictions
    predictions = pd.Series(data=predictions, index=data_test.index)
    
    # Calculate MSE
    error_mse = round(mean_squared_error(
                y_true = data_test[search],
                y_pred = predictions
            ),2)
    print(f"Test error (mse): {error_mse} \n")

    return error_mse
        
        
        
        
def SARIMA_model(train, test, order, seasonal_order, search):
    '''
    Fit SARIMA model and output Data Frame with results
    Input:
    train - data frame for training
    test - data frame for testing
    order - tuple, order for the model
    seasonal_order - tuple, seasonal order for the model
    search - string, search term
    
    Return
    Data Frame with predictions and test data
    
    '''
    # fit model
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    # make prediction             
    yhat = model_fit.predict(len(train), len(train) + len(test) - 1)
    res=pd.DataFrame({"pred":yhat, search:test.values})
        
    return res




def sarima_forecast_file_search(file, period, steps, search, plot = True):
    '''
    Read the data and forecast for chosen search term with chosen regressor
    adapted from https://joaquinamatrodrigo.github.io/skforecast/0.3/guides/autoregresive-forecaster-exogenous.html 
    Input:
    file - lowercase name of the file 
    period - last date for the data, string, format Y-m-d, maxinmum 2023-01-01
    steps - int, test split (number of weeks for test data and predicting)
    search - string, search term    
    order - tuple, order for the model
    seasonal_order - tuple, seasonal order for the model
    plot - boolean, do we want a plot
    
    Output:
    State and Search term
    Forecaster output
    Predictions vs actual plot
    MSE 
    
    
    Return:
    Data Frame with results
    '''
    # Print state and search
    print(f'\n Forecast for {file} and {search} untill {period} \n' )
    
    # Read the data
    df = pd.read_csv(f'../data/{file}.csv', parse_dates=['week'], index_col='week')
    df = df[df.index < period]
    df = df.asfreq('W-SUN')
    
    # get train and test data
    df_train = df[search][:-steps]
    df_test = df[search][-steps:]
    
    # fit the model
    df_ret = SARIMA_model(df_train, df_test, (1, 1, 1), (1, 1, 1, 12), search)
    
    # MSE
    print('MSE')
    mse = round(mean_squared_error(df_ret[search], df_ret['pred']), 2)
    print(mse)
    
    search_str = search.title()
    if search == 'mental_health':
        search_str = 'Mental Health'
    
    # plot 
    if plot == True:
        plt.figure(figsize = (10, 6))
        plt.plot(df_ret['pred'], label = 'forecast')
        plt.plot(df.index, df[search], label = 'actual')
        plt.legend()
        plt.title(f'{search_str} SARIMAX forecasting compared to actual', size=20)
        plt.ylabel(search_str, size=20);
    
    return mse


def dict_diff(dictionary):
    '''
    Find difference between similar keys in dictionary
    Input - dictionary
    Output - dictionary with value differences between simiral keys
    '''
    defferenses = {}
    for key, val in dictionary.items():
        for k,v in dictionary.items():
            if (k != key) and (k in key):
                defferenses[k] = round((val - v),2 )
    return defferenses




def sarimax_forecast_file_search(file, period, steps, search, plot = True):
    '''
    Read the data and forecast for chosen search term with chosen regressor
    adapted from https://joaquinamatrodrigo.github.io/skforecast/0.3/guides/autoregresive-forecaster-exogenous.html 
    Input:
    file - lowercase name of the file 
    period - last date for the data, string, format Y-m-d, maxinmum 2023-01-01
    steps - int, test split (number of weeks for test data and predicting)
    search - string, search term    
    order - tuple, order for the model
    seasonal_order - tuple, seasonal order for the model
    plot - boolean, do we want a plot
    
    Output:
    State and Search term
    Forecaster output
    Predictions vs actual plot
    MSE 
    
    Return:
    Data Frame with results
    '''
    # Print state and search
    print(f'\n Forecast for {file} and {search} untill {period} \n' )
    
    # Read the data
    df = pd.read_csv(f'../data/{file}.csv', parse_dates=['week'], index_col='week')
    df = df[df.index < period]
    df = df.asfreq('W-SUN')
    
    # get train and test data
    df_train = df[[search, 'stay_at_home', 'gatherings_banned', 'business_closures', 'travel_restrictions']][:-steps]
    df_test = df[[search, 'stay_at_home', 'gatherings_banned', 'business_closures', 'travel_restrictions']][-steps:]
    
    # fit the model
    df_ret = SARIMAX_model(df_train, df_test, (1, 1, 1), (1, 1, 1, 12), search)
    
    # MSE
    print('MSE')
    mse = round(mean_squared_error(df_ret[search], df_ret['pred']), 2)
    print(mse)
    
    search_str = search.title()
    if search == 'mental_health':
        search_str = 'Mental Health'
    
    # plot 
    if plot == True:
        plt.figure(figsize = (10, 6))
        plt.plot(df_ret['pred'], label = 'forecast')
        plt.plot(df.index, df[search], label = 'actual')
        plt.legend()
        plt.title(f'{search_str} SARIMAX forecasting compared to actual', size=20)
        plt.ylabel(search_str, size=20);
    
    return mse



def SARIMAX_model(train, test, order, seasonal_order, search):
    '''
    Fit SARIMAX model and output Data Frame with results
    Input:
    train - data frame for training
    test - data frame for testing
    order - tuple, order for the model
    seasonal_order - tuple, seasonal order for the model
    search - string, search term
    
    Return
    Data Frame with predictions and test data
    
    '''
    # fit model
    model = SARIMAX(train.drop(columns = ['stay_at_home', 'gatherings_banned', 'business_closures', 'travel_restrictions']), exog=train[['stay_at_home', 'gatherings_banned', 'business_closures', 'travel_restrictions']], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(train), len(train) + len(test) - 1, exog=test[['stay_at_home', 'gatherings_banned', 'business_closures', 'travel_restrictions']].values)
      
    res=pd.DataFrame({'pred':yhat, search:test[search].values})

    return res



def plot_MSE_difference(rmf_mse_dict, sarimax_mse_dict, restrictions, period, ylim):
    '''
    Plot the diffetences between the models
    '''
    
    fig, ax = plt.subplots(1,2, figsize=(14,7))
    fig.suptitle(f'MSE differences for models with and without exogenous features for {restrictions} restricted states\n', fontsize=20)
    
    # Plot Recursive multi-step forecasting
    cols = ['darkolivegreen' if (x < 0) else 'burlywood' for x in rmf_mse_dict.values()]
    ax[0].set_title('Recursive multi-step forecasting', fontsize=16)
    ax[0].bar(range(len(rmf_mse_dict)), list(rmf_mse_dict.values()), tick_label=list(rmf_mse_dict.keys()), color = cols)
    ax[0].set_ylim(ylim)
    ax[0].set_ylabel('MSE difference', fontsize=15)
    ax[0].set_xticklabels(['Depression', 'Anxiety', 'Addiction', 'Counceling', 'Mental Health'] )
    ax[0].xaxis.set_tick_params(labelsize=15, rotation = 15)

    # Plost SARIMAX
    cols_1 = ['darkolivegreen' if (x < 0) else 'burlywood' for x in sarimax_mse_dict.values()]
    ax[1].bar(range(len(sarimax_mse_dict)), list(sarimax_mse_dict.values()), tick_label=list(sarimax_mse_dict.keys()), color = cols_1)
    ax[1].set_title('SARIMAX', fontsize=16)
    ax[1].set_ylim(ylim)
    ax[1].xaxis.set_tick_params(labelsize=15)
    ax[1].set_xticklabels(['Depression', 'Anxiety', 'Addiction', 'Counceling', 'Mental Health'] )
    ax[1].xaxis.set_tick_params(labelsize=15, rotation = 15)
    
    
    # Save for the presentation
    plt.tight_layout()
    plt.savefig(f'../images/{restrictions}_mse_diff_{period}.jpeg');