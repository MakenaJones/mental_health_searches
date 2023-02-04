# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 4: Mental Health Google Searches & State Covid Restrictions

---
### Problem Statement
Our data science group was hired by the [National Institute on Mental Health](https://www.nimh.nih.gov/) to evaluate the effect of the Covid-19 pandemic, particularly with regards to the effect of restrictions put in place in response to the pandemic. With the huge range of restrictions across states, we decided to approach this problem by looking at Google Trends data for different mental health related terms in a variety of states with a variety of restrictions in place before, during and after the pandemic.

Can we use time series data with covid restrictions as exogenous features to predict mental health related google searches?  By modelling on both groups of states with strict restrictions and those with less strict restrictions, will models including the various state restrictions create better forecasts for the various mental health related google search terms than models without?

We will evaluate our models by comparing the Mean Standard Error (MSE) of models 

---

### Table of Contents

1. [Data Preparation](../code/01_Data_preparation.ipynb)



2. [EDA](../code/02_EDA.ipynb)


3. [Autocorrelation Trend Detection]

Quick search of different models of ARIMA for different search terms. Looking for changepoints using greykite based on various lockdown periods. Do not appear to be tied to restriction timeframe in a meaningful way.

4. [Models for the Start of the Covid-19 Pandemic](../code/04_Start_COVID_ForecasterAutoreg_SARIMAX.ipynb)



5. [Models for the Middle of the Covid-19 Pandemic](../code/05_Middle_COVID_ForecasterAutoreg_SARIMAX.ipynb)



6. [Models for the End of the Covid-19 Pandemic](../code/06_End_COVID_ForecasterAutoreg_SARIMAX.ipynb)



7. [Vector Autoregression](../code/07_Vector_Autoregression.ipynb)



8. [TBATS](../code/08_TBATS.ipynb)



9. [SARIMA](../code/09_SARIMA.ipynb)



10. [Greykite](../code/10_Greykite.ipynb)


---
### Data
---
Using google trends , we gathered google search data related to 5 mental health terms (anxiety, depression, addiction, counseling and mental health) from January 2018 - January 2023 for 10 states (Alaska, Arizona, California, Florida, Hawaii, Massachussetts, New York, South Dakoda, Texas and Washington), and combined it with data on what state-mandated restrictions were in place during the period.

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
    * [Covid State Restrictions](../sources/covid_restrictions.txt)

### Datasets
|Dataset|Description|
|---|---|
|[alaska.csv]('../data/alaska.csv')| Google search data related to 5 mental health terms (anxiety, depression, addiction, counseling and mental health) combined with data on state-mandated Covid-19 restrictions for the state of Alaska from January 2018 - January 2023.
|[all_states.csv]('../data/all_states.csv')| Concatenated all 10 state datasets. Added categorical column for whether state belonged to 'least restricted' or 'most restricted' group of states.
|[arizona.csv]('../data/arizona.csv')| Google search data combined with data on state-mandated Covid-19 restrictions for the state of Arizona from January 2018 - January 2023.
|[california.csv]('../data/california.csv')| Google search data combined with data on state-mandated Covid-19 restrictions for the state of California from January 2018 - January 2023.
|[combined_states.csv]('../data/all_states.csv')| Combined data of all the states with values of each column referring to the mean of values for that variable across all states for that week.
|[florida.csv]('../data/florida.csv')| Google search data combined with data on state-mandated Covid-19 restrictions for the state of California from January 2018 - January 2023.
|[hawaii.csv]('../data/florida.csv')| Google search data combined with data on state-mandated Covid-19 restrictions for the state of California from January 2018 - January 2023.
|[least_restricted.csv]('../data/least_restricted.csv')| Concatenated datasets for the states with the least Covid-19 restrictions in place during the pandemic: Arizona, Florida, South Dakota and Texas.
|[mass.csv]('../data/mass.csv')| Google search data combined with data on state-mandated Covid-19 restrictions for the state of Massachusetts from January 2018 - January 2023.
|[most_restricted.csv]('../data/most_restricted.csv')| Concatenated datasets for the states with the most Covid-19 restrictions in place during the pandemic: Alaska, California, Hawaii, Massachusetts, New York and Washington.
|[new_york.csv]('../data/new_york.csv')| Google search data combined with data on state-mandated Covid-19 restrictions for the state of New York from January 2018 - January 2023.
|[south_dakota.csv]('../data/south_dakota.csv')| Google search data combined with data on state-mandated Covid-19 restrictions for the state of South Dakota from January 2018 - January 2023.
|[texas.csv]('../data/texas.csv')| Google search data combined with data on state-mandated Covid-19 restrictions for the state of Texas from January 2018 - January 2023.
|[washington.csv]('../data/washington.csv')| Google search data combined with data on state-mandated Covid-19 restrictions for the state of Washington state from January 2018 - January 2023.

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
### EDA

* Grouped states into groups: 'most restricted' and 'least restricted' after examining search terms per state and realising there was too much information to make meaningful predictions. 
* Compared searches of different mental health search for the two groups of states and found, although similar, there was a noticable difference between the two groups during the COVID-19 pandemic, in the middle of 2020, particularly for the search term 'mental health'.
* Checked that data was stationary for both groups using the augmented Dickey-Fuller Test.
* Checked seasonality of search datas and found:
    * 'Depression' searches trend down. 
    * 'Mental health' and 'anxiety' rising sharply. 
    * 'Mental health' started to rise in the middle of 2021 after most of the restrictions were lifted. 
    * 'Anxiety' searches had a sharp rise at the beginning of the COVID restrictions and still remains high
    * 'Counselling' and 'addiction' both had a dip around the end of 2020.
---
### Model Evaluation

After testing out different time series models, we picked our two best performing models, recursive multi-step forecasting and SARIMAX and explored how they performed for 3 different time periods, with and without exogenous variables.

I. Forecast Models Before Pandemic

* Forecasting 'counselling' searches for the Most Restricted States benefited from including exogenous features (COVID Restrictions) for both SARIMA and Recursive multi-step models. 
* Forecasting mental health related search terms in Least Restricted states using recursive multi-step forecasting did not benefit from adding exogenous features (COVID Restrictions) and only improved the performance (reduced the MSE) for SARIMAX in forecasting 'depression' searches.

II. Forecast Models During Pandemic
* Both SARIMA and Recursive multi-step models for the Most Restricted States were improved when it came to forecasting 'anxiety', 'mental health' and particularly 'depression', when exogenous features were included in the models.
* The recursive multistep forecasting models for the least restricted states were improved for 'depression', 'anxiety' and particularly 'mental health' searches by including exogenous features whereas the SARIMA with exogenous features only performed better for this group of states in predicting 'addiction' searches.

III. Forecast Models After Pandemic

---
### Conclusions and Further Study

---
### Software Requirements
