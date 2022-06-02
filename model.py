"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    data = pd.DataFrame.from_dict([feature_vector_dict])

    #OUR CODE
    # replace missing values with the most frequent value in 'Valencia_pressure'
    data['Valencia_pressure'] = data['Valencia_pressure'].fillna(data['Valencia_pressure'].mode()[0])
    # drop columns
    df = data.drop(columns = ['Valencia_wind_deg','Seville_pressure'])

    # create subsets of training data in terms of weather feature
    df_wind_speed = df[['Madrid_wind_speed','Valencia_wind_speed','Bilbao_wind_speed','Barcelona_wind_speed','Seville_wind_speed']]
    df_wind_deg = df[['Bilbao_wind_deg','Barcelona_wind_deg']]
    df_humidity = df[['Seville_humidity','Madrid_humidity','Valencia_humidity']]
    df_rain = df[['Bilbao_rain_1h','Barcelona_rain_1h','Seville_rain_1h','Seville_rain_3h','Madrid_rain_1h','Barcelona_rain_3h']]
    df_clouds_all = df[['Bilbao_clouds_all','Seville_clouds_all','Madrid_clouds_all']]
    df_pressure = df[['Barcelona_pressure','Bilbao_pressure','Valencia_pressure','Madrid_pressure']]
    df_snow = df[['Bilbao_snow_3h','Valencia_snow_3h']]
    df_weather_id = df[['Madrid_weather_id','Barcelona_weather_id','Seville_weather_id','Bilbao_weather_id']]
    df_temp_min = df[['Valencia_temp_min','Bilbao_temp_min','Barcelona_temp_min','Seville_temp_min','Madrid_temp_min']]
    df_temp = df[['Barcelona_temp','Bilbao_temp','Madrid_temp','Seville_temp','Valencia_temp']]
    df_temp_max = df[['Barcelona_temp_max','Bilbao_temp_max','Madrid_temp_max','Seville_temp_max','Valencia_temp_max']]

    # drop 1h rain feature data
    df_rain = df_rain.drop(columns = ['Bilbao_rain_1h','Barcelona_rain_1h','Seville_rain_1h', 'Madrid_rain_1h'])

    # create a series containing meam value of weather feature across all cities
    ave_wind_speed = df_wind_speed.mean(axis = 1)
    ave_wind_deg = df_wind_deg.mean(axis = 1)
    ave_humidity = df_humidity.mean(axis = 1)
    ave_rain = df_rain.mean(axis = 1)
    ave_clouds_all = df_clouds_all.mean(axis = 1)
    ave_pressure = df_pressure.mean(axis = 1)
    ave_snow = df_snow.mean(axis = 1)
    ave_weather_id = df_weather_id.mean(axis = 1)
    ave_temp_min = df_temp_min.mean(axis = 1)
    ave_temp = df_temp.mean(axis = 1)
    ave_temp_max = df_temp_max.mean(axis = 1)

    # creating a time feature and load_shortfall series
    time = df['time']
    load_shortfall_3h = df['load_shortfall_3h']

    # create a dictionary containing all feature series with their names and keys in a desirable order
    features_ave = {'Time': time, 'Ave_weather_id': ave_weather_id, 'Ave_wind_speed': ave_wind_speed, 'Ave_wind_deg': ave_wind_deg, 'Ave_humidity': ave_humidity, 
                'Ave_rain': ave_rain, 'Ave_clouds_all': ave_clouds_all, 'Ave_pressure': ave_pressure, 'Ave_snow': ave_snow, 'Ave_temp_min': ave_temp_min,
                'Ave_temp': ave_temp, 'Ave_temp_max': ave_temp_max, 'load_shortfall_3h': load_shortfall_3h}

    # convert dictionary of feature series into dataframe named df_train_clean
    data_clean = pd.DataFrame(features_ave)

    # convert time data type to datetime
    data_clean['Time'] = pd.to_datetime(data_clean['Time'])

    # create new features
    data_clean['Hour_of_day'] = data_clean['Time'].dt.hour # add hour of day feature
    data_clean['Day_of_year'] = data_clean['Time'].dt.day_of_year # add day of the year feature
    data_clean['Week_of_year'] = data_clean['Time'].dt.isocalendar().week # add week of year feature

    # engineer existing fetures

    # create a list of features we need to bring forward
    first_columns = ['load_shortfall_3h', 'Time', 'Hour_of_day', 'Day_of_year', 'Week_of_year']

    # create the order of columns
    new_columns = first_columns + [col for col in data_clean.columns if col not in first_columns] 

    # create a new dataframe named 'df_train' with the new column index 
    data = data_clean.reindex(columns = new_columns)  

    # define outlier detector function
    def handle_outliers(dataframe, col):
        '''
        define function which takes as argument a dataframe and a column
        as input, calculates boundaries for outliers and replaces each outlier
        by it closest boundary value, and then returns the
        '''
        iqr = dataframe[col].quantile(0.75) - dataframe[col].quantile(0.25)    # calculate interquantile range for column
        tail = dataframe[col].quantile(0.25) - (iqr*1.5)                      # set lower boundary
        head= dataframe[col].quantile(0.75) + (iqr*1.5)                      # set upper boundary
        dataframe.loc[dataframe[col] > head, col] = head                    # detect and replace each outlier with nearest boundary
        dataframe.loc[dataframe[col] < tail, col] = tail    
        return dataframe

    # call handle_outliers for features with high kurtosis 
    handle_outliers(data, 'Ave_pressure')
    handle_outliers(data, 'Ave_rain')
    handle_outliers(data, 'Ave_snow')
    handle_outliers(data, 'Ave_wind_speed')
    handle_outliers(data, 'Ave_weather_id')
    
    return data

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
