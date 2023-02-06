# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 4: Mental Health Google Searches & State Covid Restrictions

---
## Problem Statement
Our data science group was hired by the [National Institute on Mental Health](https://www.nimh.nih.gov/) to evaluate the effect of the Covid-19 pandemic, particularly with regards to the effect of restrictions put in place in response to the pandemic. With the huge range of restrictions across states, we decided to approach this problem by looking at Google Trends data for different mental health related terms in a variety of states with a variety of restrictions in place before, during and after the pandemic.

Can we use time series data with covid restrictions as exogenous features to predict mental health related google searches?  By modelling on both groups of states with strict restrictions and those with less strict restrictions, will models including the various state restrictions create better forecasts for the various mental health related google search terms than models without?

We will evaluate our models by comparing the Mean Standard Error (MSE) of models with and without exogenous features, for both groups of states with less restrictions and those with more restrictions enforced during the pandemic.

Will the models see the changing search pattern after restrictions were enforced?

---

## Table of Contents

1. [Data Preparation](https://github.com/MakenaJones/mental_health_searches/blob/main/code/01_Data_preparation.ipynb) : Read and combine data from soures. Group data for EDA and Modeling.

2. [EDA](https://github.com/MakenaJones/mental_health_searches/blob/main/code/02_EDA.ipynb) :  Explore data and determine it is both stationary and exhibits seasonality.

3. [Autocorrelation Trend Detection](https://github.com/MakenaJones/mental_health_searches/blob/main/code/03_Autocorrelation_Trend_Detection.ipynb) : Quick search of different models of ARIMA for different search terms. Also looked for changepoints using greykite based on various lockdown periods.

4. [Models for the Start of the Covid-19 Pandemic](https://github.com/MakenaJones/mental_health_searches/blob/main/code/04_Start_COVID_ForecasterAutoreg_SARIMAX.ipynb) : Explore and evaluate the impact of COVID-19 restrictions on mental health related searches by exploring two different models, Recursive Multistep Forecasting and SARIMAX, with and without COVID-19 Restrictions as Exogenous features, for the time period leading up to and including the beginning of the COVID-19 Pandemic.

5. [Models for the Middle of the Covid-19 Pandemic](https://github.com/MakenaJones/mental_health_searches/blob/main/code/05_Middle_COVID_ForecasterAutoreg_SARIMAX.ipynb) : Explore and evaluate the same models as in notebook 4, but this time for the time period during the height of the COVID-19 Pandemic.

6. [Models for the End of the Covid-19 Pandemic](https://github.com/MakenaJones/mental_health_searches/blob/main/code/06_end_COVID_ForecasterAutoreg_SARIMAX.ipynb) : Explore and evaluate the same models as in notebooks 4 and 5, but this time for the time period nearing the end of the COVID-19 Pandemic.

7. [Vector Autoregression](https://github.com/MakenaJones/mental_health_searches/blob/main/code/07_Vector_Autoregression.ipynb) : Forecasted on all searches together to find the best models by finding changes in trends after COVID-19 restrictions started.

8. [TBATS](https://github.com/MakenaJones/mental_health_searches/blob/main/code/08_TBATS.ipynb) : Forecasted with TBATS to find the best models by finding changes in trend after COVID-19 restrictions started.

9. [SARIMA](https://github.com/MakenaJones/mental_health_searches/blob/main/code/09_SARIMA.ipynb) : Forecasted with SARIMA, tuning and fitting on data before COVID-19 and checked the difference in the forecast and actual values for the beginning of the COVID-19 restrictions.

10. [Greykite](https://github.com/MakenaJones/mental_health_searches/blob/main/code/10_Greykite.ipynbb) : Forecasted with Greykite time series model, tuning and fitting on data before COVID-19 and checked the difference in the forecast and actual values for the beginning of the COVID-19 restrictions.

11. [Prophet](https://github.com/MakenaJones/mental_health_searches/blob/main/code/11_Prophet.ipynb) : Forecasted with Prophet using covid restrictions inputed as one-time holidays.

---
## Data

Using google trends, we gathered google search data related to 5 mental health terms (anxiety, depression, addiction, counseling and mental health) from January 2018 - January 2023 for 10 states (Alaska, Arizona, California, Florida, Hawaii, Massachussetts, New York, South Dakoda, Texas and Washington), and combined it with data on what state-mandated restrictions were in place during the period.

### Data sources

* Google trends:
    * [Alaska](https://trends.google.com/trends/explore?date=2018-01-01%202023-01-01&geo=US-AK&q=depression,anxiety,addiction,counseling,mental%20health)
    * [Arizona](https://trends.google.com/trends/explore?date=2018-01-01%202023-01-01&geo=US-AZ&q=depression,anxiety,addiction,counseling,mental%20health)
    * [California](https://trends.google.com/trends/explore?date=2018-01-01%202023-01-01&geo=US-CA&q=depression,anxiety,addiction,counseling,mental%20health)
    * [Florida](https://trends.google.com/trends/explore?date=2018-01-01%202023-01-01&geo=US-FL&q=depression,anxiety,addiction,counseling,mental%20health)
    * [Hawaii](https://trends.google.com/trends/explore?date=2018-01-01%202023-01-01&geo=US-HI&q=depression,anxiety,addiction,counseling,mental%20health)
    * [Massachusetts](https://trends.google.com/trends/explore?date=2018-01-01%202023-01-01&geo=US-MA&q=depression,anxiety,addiction,counseling,mental%20health)
    * [New York](https://trends.google.com/trends/explore?date=2018-01-01%202023-01-01&geo=US-NY&q=depression,anxiety,addiction,counseling,mental%20health)
    * [South Dakota](https://trends.google.com/trends/explore?date=2018-01-01%202023-01-01&geo=US-SD&q=depression,anxiety,addiction,counseling,mental%20health)
    * [Texas](https://trends.google.com/trends/explore?date=2018-01-01%202023-01-01&geo=US-TX&q=depression,anxiety,addiction,counseling,mental%20health)
    * [Washington](https://trends.google.com/trends/explore?date=2018-01-01%202023-01-01&geo=US-WA&q=depression,anxiety,addiction,counseling,mental%20health)


* Sources for timelines of different state-mandated Covid-19 restrictions:
    * [Covid State Restrictions](https://github.com/MakenaJones/mental_health_searches/blob/main/Sources/covid_restrictions.txt)

### Datasets
|Dataset|Description|
|---|---|
|[alaska.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/alaska.csv)| Google search data related to 5 mental health terms (anxiety, depression, addiction, counseling and mental health) combined with data on state-mandated Covid-19 restrictions for the state of Alaska from January 2018 - January 2023.
|[all_states.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/all_states.csv)| Concatenated all 10 state datasets. Added categorical column for whether state belonged to 'least restricted' or 'most restricted' group of states.
|[arizona.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/arizona.csv)| Google search data combined with data on state-mandated Covid-19 restrictions for the state of Arizona from January 2018 - January 2023.
|[california.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/california.csv)| Google search data combined with data on state-mandated Covid-19 restrictions for the state of California from January 2018 - January 2023.
|[combined_states.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/combined_states.csv)| Combined data of all the states with values of each column referring to the mean of values for that variable across all states for that week.
|[florida.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/florida.csv)| Google search data combined with data on state-mandated Covid-19 restrictions for the state of California from January 2018 - January 2023.
|[hawaii.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/hawaii.csv)| Google search data combined with data on state-mandated Covid-19 restrictions for the state of California from January 2018 - January 2023.
|[least_restricted.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/least_restricted.csv)| Concatenated datasets for the states with the least Covid-19 restrictions in place during the pandemic: Arizona, Florida, South Dakota and Texas.
|[mass.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/mass.csv)| Google search data combined with data on state-mandated Covid-19 restrictions for the state of Massachusetts from January 2018 - January 2023.
|[most_restricted.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/most_restricted.csv)| Concatenated datasets for the states with the most Covid-19 restrictions in place during the pandemic: Alaska, California, Hawaii, Massachusetts, New York and Washington.
|[new_york.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/new_york.csv)| Google search data combined with data on state-mandated Covid-19 restrictions for the state of New York from January 2018 - January 2023.
|[south_dakota.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/south_dakota.csv)| Google search data combined with data on state-mandated Covid-19 restrictions for the state of South Dakota from January 2018 - January 2023.
|[texas.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/texas.csv)| Google search data combined with data on state-mandated Covid-19 restrictions for the state of Texas from January 2018 - January 2023.
|[washington.csv](https://github.com/MakenaJones/mental_health_searches/blob/main/data/washington.csv)| Google search data combined with data on state-mandated Covid-19 restrictions for the state of Washington state from January 2018 - January 2023.

### Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|addiction|Variable|All datasets in data folder.| Numbers represent search interest in the term 'addiction' relative to the highest point on the chart for the given region and time (in our case, for the given state and week) on Google Trends. A value of 100 would be peak popularity.
|anxiety|Variable|All datasets in data folder.| Numbers represent search interest in the term 'anxiety' relative to the highest point on the chart for the given region and time (in our case, for the given state and week) on Google Trends. A value of 100 would be peak popularity.
|business_closures|Variable|All datasets in data folder.| Binary variable with a value of 0 if state did not mandate significant business closures (including restaurants, retail, etc.) in the week leading up to the time period, and a value of 1 if such restrictions were indeed in place.
|counselling|Variable|All datasets in data folder.| Numbers represent search interest in the term 'counseling' relative to the highest point on the chart for the given region and time (in our case, for the given state and week) on Google Trends. A value of 100 would be peak popularity.
|depression|Variable|All datasets in data folder.| Numbers represent search interest in the term 'depression' relative to the highest point on the chart for the given region and time (in our case, for the given state and week) on Google Trends. A value of 100 would be peak popularity.
|gatherings_closures|Variable|All datasets in data folder.| Binary variable with a value of 1 if state banned gatherings in response to the Covid-19 pandemic in the week leading up to the time period, and a value of 0 if no ban was in place.
|mask_mandates|Variable|All datasets in data folder.| Binary variable with a value of 1 if state mandated the wearing of masks in public spaces in response to the pandemic, and a value of 0 if no ban was in place.
|mental_health|Variable|All datasets in data folder.| Numbers represent search interest in the term 'mental health' relative to the highest point on the chart for the given region and time (in our case, for the given state and week) on Google Trends. A value of 100 would be peak popularity.
|state|Variable|All individual state datasets in data folder.| American state used as region for google trends and evaluating what Covid-19 restrictions were in place.
|stay_at_home|Variable|All datasets in data folder.| Binary variable with a value of 1 if state had a stay-at-home order in effect in the week leading up to the date in the week of interest and a value of 0 if no stay-at-home order was in place.
|travel_restrictions|Variable|All datasets in data folder.| Binary variable with a value of 1 if state restricted all inter-state travel in effect in the week leading up to the date in the week of interest and a value of 0 if no such restriction was in place.
|week|Variable|All datasets in data folder.| 7 day period ending in the date specified.

---
## EDA

   Started by grouping states into 2 groups: 'most restricted' and 'least restricted' after examining search terms per state and realising there was too much information to make meaningful predictions. 
   Compared searches of different mental health search for the two groups of states and found, although similar, there was a noticable difference between the two groups during the COVID-19 pandemic, in the middle of 2020, particularly for the search term 'mental health'.
   Checked that data was stationary for both groups using the augmented Dickey-Fuller Test.
   Checked seasonality and trends of search data and found:
* 'Depression' searches trend down. 
* 'Mental health' started to rise in the middle of 2021 after most of the restrictions were lifted. 
* 'Anxiety' searches had a rise at the beginning of the COVID restrictions and still remain high.
* 'Counselling' and 'addiction' both had a dip around the end of 2020.
* 'Depression' and 'mental health' have the strongest seasonality. 

---

## Model Evaluation

After testing out different time series models, we picked our two best performing models, recursive multi-step forecasting and SARIMAX and explored how they performed for 3 different time periods, with and without exogenous variables.

I. Forecast Models Before Pandemic

* Forecasting 'counseling' searches for the Most Restricted States benefited from including exogenous features (COVID Restrictions) for both SARIMA and Recursive multi-step models. 
![start_most](https://github.com/MakenaJones/mental_health_searches/blob/main/images/most_mse_diff_2020-05-30.jpeg) 

* Forecasting mental health related search terms in Least Restricted states using recursive multi-step forecasting did not benefit from adding exogenous features (COVID Restrictions) and only improved the performance (reduced the MSE) for SARIMAX in forecasting 'depression' searches.
![start_least](https://github.com/MakenaJones/mental_health_searches/blob/main/images/least_mse_diff_2020-05-30.jpeg) 

II. Forecast Models During Pandemic
* Both SARIMA and Recursive multi-step models for the Most Restricted States were improved when it came to forecasting 'anxiety'and 'depression', when exogenous features were included in the models.
![middle_most](https://github.com/MakenaJones/mental_health_searches/blob/main/images/most_mse_diff_2020-09-30.jpeg)

* The recursive multistep forecasting models for the least restricted states were improved for 'depression', 'anxiety' and particularly 'mental health' searches by including exogenous features whereas the SARIMA with exogenous features only performed better for this group of states in predicting 'addiction' searches.
![middle_least](https://github.com/MakenaJones/mental_health_searches/blob/main/images/least_mse_diff_2020-09-30.jpeg)

III. Forecast Models After Pandemic
* Most restricted states did not benefit from adding exogenous features for both SARIMAX and recursive multi-step forecasting models in any of the search terms. However, adding restrictions as exogenous features improved the performance of forecasting:
    * 'depression', 'anxiety' and 'mental health' in recursive multi-step forecasting models
    * 'mental health' searches using the SARIMAX model  
![end_most](https://github.com/MakenaJones/mental_health_searches/blob/main/images/most_mse_diff_2021-01-01.jpeg)

* In the least restricted states,forecasting 'depression' and 'counseling' searches benefited from adding COVID Restrictions as exogenous features in both SARIMAX and recursive multi-step forecasting models.
![end_least](https://github.com/MakenaJones/mental_health_searches/blob/main/images/least_mse_diff_2021-01-01.jpeg)

We fitted SARIMA and Greykite on data before COVID-19 and checked the difference in the forecast and actual values for the beginning of the COVID-19 restrictions. SARIMA and Greykite had slightly different predictions for the beginning of COVID-19, but they both over-predicted counseling searches.
![SARIMA](https://github.com/MakenaJones/mental_health_searches/blob/main/images/forecasting_sarima_counselling.jpeg)

Prophet time series modelling, which performed well in predicting some mental health searches, did not perform better by adding COVID-19 restrictions as one-time holidays during the time periods at the beginning, middle and end of the pandemic. 

---
## Conclusions
Regarding the first part of our problem statement, on whether time series models including the various state restrictions create better forecasts for the various mental health related google search terms than models without, the answer is for certain search terms during certain time periods. More specifically:
* Mental health had an increase in searches towards the end of the pandemic in all states (regardless of restrictions in place)
* Forecasting 'counseling' in beginning of pandemic using restrictions as exogenous features improved the performance of both models (multistep/SARIMAX) for most restricted states.
* In the middle of the pandemic, SARIMAX and Recursive multi-step models for the most Restricted States were improved when it came to forecasting 'anxiety', 'mental health' and particularly 'depression', when exogenous features were included in the models.
* At the end of the pandemic, forecasting 'depression' and 'counseling' searches benefited from adding COVID Restrictions as exogenous features in both SARIMAX and recursive multi-step forecasting models for least restricted states.

To answer the second part of our problem statement, regarding whether models see the changing search pattern after restrictions were enforced, the answer is only for 'counseling' searches at the beginning of the Covid-19 pandemic. More specifically:
* SARIMA and Greykite overpredicted counseling searches at the beginning of the Covid-19 restrictions compared to normal for this period of time (as they actually ended up plummeting).

---
## Further Study
 If wanting to isolate the effect of the restrictions themselves, we would have to incorporate other factors that were occuring concurrently before, during and after the pandemic into our models.
 Would be interesting to examine if actual patient numbers/calls to health lines/requests for telehealth services changed for counseling/mental health related services during the periods we examined since we were only looking at Google searches. 

---
## Software Requirements

For this project, we imported pandas, listdir, matplotlib, numpy, seaborn, statsmodels, greykite (note: have to create seperate environment - use this [guide](https://linkedin.github.io/greykite/installation)), datetime, warnings, collections, sklearn, plotly, itertools, tqdm, prophet, math, multiprocessing and skforecast.