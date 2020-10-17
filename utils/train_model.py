"""
    Simple file to create a Sklearn model for deployment in our API
    Author: Explore Data Science Academy
    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.
"""

# Dependencies
import pandas as pd
import pickle
from xgboost import plot_importance
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')
riders = pd.read_csv('data/riders.csv')
train = train.merge(riders, how='left', on='Rider Id')

#Data Preprocessing

def _preprocess_data(data):

    data = data.copy()
    data.columns = [col.replace(" ","_") for col in data.columns]
    Categorical_Data = data.select_dtypes(include=['object'])
    le = LabelEncoder()
    encoded_categorical_Data = Categorical_Data.apply(lambda x: le.fit_transform(x))
    Numeric_Data = data._get_numeric_data()
    data_encoded = pd.concat([encoded_categorical_Data, Numeric_Data], axis=1)
    cols = ['Arrival_at_Destination_-_Time' , 'Arrival_at_Destination_-_Day_of_Month',
            'Arrival_at_Destination_-_Weekday_(Mo_=_1)' ,'Time_from_Pickup_to_Arrival',
            'Order_No','User_Id','Precipitation_in_millimeters','Temperature']

    x = [i for i in cols if i in data_encoded.columns]

    clean_data = data_encoded[:len(data)].drop(x,axis =1)
    predict_vector = clean_data

    return predict_vector


train =  _preprocess_data(train)


#Train the model 
y_train = train[['Time from Pickup to Arrival']]
X_train = _preprocess_data(train)

# Fit model

xgb_model = xgb.XGBRegressor(colsample_bytree=0.7,
                              learning_rate=0.1,
                              max_depth=3,
                              min_child_weight=3,
                              n_estimators=500,
                              objective='reg:squarederror',
                              subsample=0.7)
print ("Training Model...")
xgb_model.fit(X_train,y_train)

# Pickle model for use within our API
save_path = '../trained-models/xgb_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(xgb_model, open(save_path,'wb'))
