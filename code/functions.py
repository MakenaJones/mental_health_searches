import pandas as pd
import warnings
import matplotlib.pyplot as plt
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
    
    if error_mse < 100:
        print(f"Test error (mse): {error_mse} \n")
        
        
        print('Feature Importances')
        print(forecaster.get_feature_importance())
        
        if plot == True:
            # Plot predictions vs actual
            fig, ax=plt.subplots(figsize=(11, 6))
            data_train[search].plot(ax=ax, label='train')
            data_test[search].plot(ax=ax, label='test')
            predictions.plot(ax=ax, label='predictions')
            ax.legend()
            ax.set_title(f'Forecast for {file} and {search}', size = 20)
            ax.set_ylabel(f'{search}', size = 20)
            ax.set_xlabel('Week', size = 20);
            plt.plot()
        
        return forecaster
    else:
        print(f"Test error (mse): {error_mse} \n")
        
        
        
        
        
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
    
    if error_mse < 100:
        print(f"Test error (mse): {error_mse} \n")
        
        
        print('Feature Importances')
        print(forecaster.get_feature_importance())
        
        if plot == True:
            # Plot predictions vs actual
            fig, ax=plt.subplots(figsize=(11, 6))
            data_train[search].plot(ax=ax, label='train')
            data_test[search].plot(ax=ax, label='test')
            predictions.plot(ax=ax, label='predictions')
            ax.legend()
            ax.set_title(f'Forecast for {file} and {search}', size = 20)
            ax.set_ylabel(f'{search}', size = 20)
            ax.set_xlabel('Week', size = 20);
            plt.plot()
        
        return forecaster
    else:
        print(f"Test error (mse): {error_mse} \n")