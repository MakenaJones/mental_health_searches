import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from skforecast.ForecasterAutoreg import ForecasterAutoreg

from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error




def forecast_file_search(file, period, steps, search, regressor, lags, plot = True):
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
    error_mse = mean_squared_error(
                y_true = data_test[search],
                y_pred = predictions
            )
    
    
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

    return forecaster
    
        
        
        
        
        
def forecast_file_search_without_exogin(file, period, steps, search, regressor, lags, plot = True):
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
    error_mse = mean_squared_error(
                y_true = data_test[search],
                y_pred = predictions
            )
    
    
    print(f"Test error (mse): {error_mse} \n")

    return forecaster
        
        
        
        
        
def plot_resiriction_importances(data, search, ylim, time):
    '''
    Plot side by side plots of Restriction importsnces for forecasting for Most and Least restricted states
    Input:
    data - dattaframe of all COVID 19 restrictions feature importances
    search - str, search term lowercase
    ylim - list, limits for y-axis 
    time - string, for the file name of the saved graphs
    
    Outputs:
    Side by side plot of restriction importances for forecasting
    Saves figure to jpeg file
    '''
    search_str = search.title()
    if search == 'mental_health':
        search_str = 'Mental Health'
    
    #use orange for bar with max value and grey for all other bars
    cols_most = ['firebrick' if (x > 0) else 'steelblue' for x in data[f'{search}_most']]
    
    fig, ax = plt.subplots(1,2, figsize=(20,7))

    fig.suptitle(f'{search_str} COVID-19 Restrictions importances for forecasting \n', fontsize=20)
    
    sns.barplot(data = data, x = 'feature', y = f'{search}_most', ax=ax[0], palette=cols_most)
    ax[0].set_title('Most restricted states', fontsize=20)
    ax[0].set_xlabel('COVID-19 Restriction', fontsize=15)
    ax[0].xaxis.set_tick_params(labelsize=15, rotation=15)
    ax[0].set_xticks(np.arange(5), ['Depression', 'Anxiety', 'Addiction', 'Counselling', 'Mental Health'])
    ax[0].set_ylabel('Restrictions importances', fontsize=20)
    ax[0].yaxis.set_tick_params(labelsize=15)
    ax[0].set_ylim(ylim)
    

    cols_least = ['firebrick' if (x > 0) else 'steelblue' for x in data[f'{search}_most']]
    sns.barplot(data = data, x = 'feature', y = f'{search}_least', ax=ax[1], palette=cols_least)
    ax[1].set_title('Least restricted states', fontsize=20)
    ax[1].set_xlabel('COVID-19 Restriction', fontsize=15)
    ax[1].set_xticks(np.arange(5), ['Depression', 'Anxiety', 'Addiction', 'Counselling', 'Mental Health'])
    ax[1].xaxis.set_tick_params(labelsize=15, rotation=15)
    ax[1].set_ylabel('Restrictions importances', fontsize=20)
    ax[1].yaxis.set_tick_params(labelsize=15)
    ax[1].set_ylim(ylim)

    # Save for the presentation 
    plt.tight_layout()
    plt.savefig(f'../images/{search_str} COVID-19 Restrictions importances {time}.jpeg');