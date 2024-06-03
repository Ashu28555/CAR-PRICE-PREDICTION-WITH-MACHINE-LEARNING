# CAR-PRICE-PREDICTION-WITH-MACHINE-LEARNING
# Task  : CAR PRICE PREDICTION WITH MACHINE LEARNING
### The price of a car depends on a lot of factors like the goodwill of the brand of the car, features of the car, horsepower and the mileage it gives and many more. Car price prediction is one of the major research areas in machine learning. So if you want to learn how to train a car price prediction model then this project is for you.

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

### Load the Dataset

data = pd.read_csv("car data.csv")

data

data.info()

### Data cleaning and preprocessing

data.isna().sum()

data.duplicated().sum()

data.drop_duplicates(inplace=True)

data.describe()

data['Fuel_Type'].value_counts()

data['Selling_type'].value_counts()

data['Transmission'].value_counts()

data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})
data['Selling_type'] = data['Selling_type'].map({'Dealer':0,'Individual':1})
data['Transmission'] = data['Transmission'].map({'Manual':0,'Automatic':1})

data.head()

data.info()

### EDA

sns.pairplot(data)

x = data.drop(columns=['Car_Name','Selling_Price'],axis =1)

x

y = data['Selling_Price']

y

### Splitting the dataset into testing and training

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train

x_test

### Train the dataset using Linear Regression Model

model = LinearRegression()

model.fit(x_train,y_train)

pred = model.predict(x_test)
pred

sns.regplot(x=y_test,y=pred)
sns.regplot(x=y_test,y=pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual Vs Predicted Selling Price")
plt.title("Actual Vs Predicted Selling Price")

### Model Evaluation

mse = mean_squared_error(y_test,pred)
mse

r2 = r2_score(y_test,pred)
r2

d1 = x.iloc[[100]]
d1

p1 = model.predict(d1)
p1

y.iloc[[20]]

### Run the model

yr = int(input('Enter the year: '))
pp = float(input('Enter the Present Price: '))
dr = int(input('Enter the Driven Kms: '))
ft = int(input('Enter the Fuel Type (0: Petrol, 1: Diesel, 2: CNG): '))
st = int(input('Enter the Selling Type (0: Dealer, 1: Individual): '))
t = int(input('Enter the Transmission (0: Manual, 1: Automatic): '))
o = int(input('Enter the Number of owners: '))
new_data = [[yr,pp,dr,ft,st,t,o]]
new_pred = model.predict(new_data)
print("The selling Price of car as per the specifications are :",new_pred[0])

## The End

