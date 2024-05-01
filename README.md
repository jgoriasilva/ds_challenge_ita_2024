# Data Science Challenge at EEF ITA 2024

This repository contains all the code I used for the Data Science Challenge at EEF ITA 2024.

## The competition

The Data Science Challenge at EEF ITA is a student data science competition held annually at the Instituto Tecnologico de Aeronautica (ITA), in Brazil, since 2019.

In this edition, the challenge involved building a machine learning model to predict waits on commercial flights.

More information about the challenge can be found at [its Kaggle page](https://www.kaggle.com/competitions/data-science-challenge-at-eef-2024/overview).

## The problem

The problem consists of predicting the presence of waits for commercial flights to and from the 12 main aerodromes in Brazil.

The available databases included information related to geolocation, satellites, weather stations and climate predictors.

Since meteorological phenomena have a strong influence on the punctuality of a flight, meteorological data was provided from 3 sources:

- METAR (Meteorological Aerodrome Report)
- Meteorological satellite images
- METAF (Terminal Aerodrome Forecast)
The METAR and METAF bases provide telemetry data from weather stations in the aerodrome region, such as temperature, wind speed, humidity, etc.

Information about runway marking changes was also available, including forecasts and historical data.

## My solution

I developed a solution using both LightGBM and XGBoost models. A considerable work was done on exploring and visualizing the data, as well as tuning the hyperparameters of the models.

I also used adversarial validation in my solution, which allow us to deal with the problem of model drift. Information about this is detailed in the *adversarial.ipynb* notebook.
