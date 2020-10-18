"""

    Simple Script to test the API once deployed

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located at the root of this repo for guidance on how to use this
    script correctly.
    ----------------------------------------------------------------------

    Description: This file contains code used to formulate a POST request
    which can be used to develop/debug the Model API once it has been
    deployed.

"""

# Import dependencies
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json

# Load data from file to send as an API POST request.
# We prepare a DataFrame with the public test set + riders data
# from the Zindi challenge.
test = pd.read_csv('data/test_data.csv')
riders = pd.read_csv('data/riders.csv')
test = test.merge(riders, how='left', on='Rider Id')


#Preprocess data

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
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ----------------------------------------------------------------------
    
    #Preprocessing of loaded data 
    feature_vector_df.columns = [col.replace(" ","_") for col in feature_vector_df.columns]
   
    Categorical_Data = feature_vector_df.select_dtypes(include=['object'])
    le = LabelEncoder()
    encoded_categorical_Data = Categorical_Data.apply(lambda x: le.fit_transform(x))
    Numeric_Data = feature_vector_df._get_numeric_data()
    data_encoded = pd.concat([encoded_categorical_Data, Numeric_Data], axis=1)
    
    cols = ['Arrival_at_Destination_-_Time' , 'Arrival_at_Destination_-_Day_of_Month',
            'Arrival_at_Destination_-_Weekday_(Mo_=_1)' ,'Time_from_Pickup_to_Arrival',
            'Order_No','User_Id','Precipitation_in_millimeters','Temperature']
    x = [i for i in cols if i in data_encoded.columns]
    clean_data = data_encoded[:len(data)].drop(x,axis =1) 
    predict_vector = clean_data
    
    # ------------------------------------------------------------------------

    return predict_vector.to_json()

test = _preprocess_data(test)

# Convert our DataFrame to a JSON string.
# This step is necessary in order to transmit our data via HTTP/S
feature_vector_json = test.iloc[1].to_json()

# Specify the URL at which the API will be hosted.
# NOTE: When testing your instance of the API on a remote machine
# replace the URL below with its public IP:

# url = 'http://{public-ip-address-of-remote-machine}:5000/api_v0.1'
url = 'http://172-31-45-195:5000/api_v0.1'

# Perform the POST request.
print(f"Sending POST request to web server API at: {url}")
print("")
print(f"Querying API with the following data: \n {test.iloc[1].to_list()}")
print("")
# Here `api_response` represents the response we get from our API
api_response = requests.post(url, json=feature_vector_json)

# Display the prediction result
print("Received POST response:")
print("*"*50)
print(f"API prediction result: {api_response.json()[0]}")
print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
print("*"*50)
