import pandas as pd  
import numpy as np
import nltk
import matplotlib.pyplot as plt   
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor


dataset=pd.read_csv("C:/Users/VIVEK REDDY S/Desktop/courseraexcel.csv")

print(dataset.shape)


print(dataset.describe())


print(dataset.isnull().any())


dataset=dataset.dropna()

#dataset=dataset.reset_index(drop=True)

print(dataset.shape)

df=pd.DataFrame(dataset,columns=['area_type','size','bath','total_sqft', 'balcony', 'price'])

print(dataset.shape)
X=df[['size','total_sqft']].values
y=df['price'].values
print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = MLPRegressor(hidden_layer_sizes=(100,100,100), activation="relu", batch_size=10, learning_rate_init=0.001, max_iter=400, random_state=1).fit(X_train, y_train)  


y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(regressor.score(X_test,y_test))
print(df)
