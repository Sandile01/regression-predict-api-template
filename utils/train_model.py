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
from sklearn.linear_model import LinearRegression,, Ridge, Lasso

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')
riders = pd.read_csv('data/riders.csv')
train_rd = train.merge(riders, how='left', on='Rider Id')

#Exploratory Data Analysis
train_rd['Order No'].value_counts()
train_rd['Platform Type'].value_counts()
train_rd['Personal or Business'].value_counts()

plt.figure(figsize=(16,8))
sns.heatmap(train_rd.corr(),annot=True) 
plt.title('Heatmap', fontsize=20)

Categorical_Train = train_rd.select_dtypes(include=['object'])

Categorical_Train.head()

le = LabelEncoder()
encoded_categorical_Train = Categorical_Train.apply(lambda x: le.fit_transform(x))

encoded_categorical_Train.head()
Numeric_Train = train_rd._get_numeric_data()
train_encoded = pd.concat([encoded_categorical_Train, Numeric_Train], axis=1)

X = train_encoded.drop(['Time from Pickup to Arrival',
                        'Precipitation in millimeters',
                       'Temperature'], axis = 1)


y = train_encoded['Time from Pickup to Arrival']


regression = [LinearRegression(), Ridge(alpha=0.01), Lasso(alpha=0.01)]
name = ['Linear','Ridge','Lasso']

base_result = []
models = {}

for name, reg in zip(name, regression):
    print ('Team 2 at work {:s} model'.format(name))
    run_time = %timeit -q -o reg.fit(X_train, y_train)
    print('...predicting')
    y_pred = reg.predict(X_test)
    print('...evaluating')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    rss = mean_squared_error(y_test, y_pred)*len(train)
    r2 = r2_score(y_test, y_pred)
    models[name] = reg
    base_result.append([name, rmse,
                        mse, rss, r2,
                        run_time.best])
    
base_result = pd.DataFrame(base_result, columns =['Regressor', 'RSME',
                                                 'MSE', 'RSS', 'R2',
                                                 'Train Time'])
base_result.set_index('Regressor', inplace = True)















y_train = train[['Time from Pickup to Arrival']]
X_train = train[['Pickup Lat','Pickup Long',
                 'Destination Lat','Destination Long','Order No','User Id','Personal or Business','Placement - Time','Confirmation - Time','']]

# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../trained-models/sendy_simple_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))
