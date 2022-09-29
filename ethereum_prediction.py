#import libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#read in the ETH price history CSV
eth_df = pd.read_csv('~/Downloads/ETH-USD.csv')
#use the date as the index
eth_df = eth_df.set_index(pd.DatetimeIndex(eth_df['Date'].values))
#print the dataframe
print(eth_df)

#create a variable to predict the number of days in the future 
days_future = 5

#create a new column to contain the target
eth_df[str(days_future)+'_Day_Price_Forecast'] = eth_df[['Close']].shift(-days_future) 
#show the data
eth_df[['Close', ]]

#prepare data for predictive model (create the X dataset)
X = np.array(eth_df[['Close']])
X = X[:eth_df.shape[0]-days_future]
print(X)

#prepare data for predictive model (create the Y dataset)
y = np.array(eth_df[str(days_future)+'_Day_Price_Forecast'])
y = y[:-days_future]
print(y)

#split the data to set up for support vector regression
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

#implement support vector regression (SVR) to train the x and y datasets 
from sklearn.svm import SVR 
#the radial basis function kernel (rbf) computes the inner product computed between two vectors
#we set the regularization parameter to 0.0001 and a kernel coefficient (gamma) of 0.00001
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
svr_rbf.fit(x_train,y_train)

#find and print the accurary score for the model
svr_rbf_confidence = svr_rbf.score(x_test,y_test)
print('svr_rbf accuracy:', svr_rbf_confidence)

#take a look at the predicted values from our regression model to compare to the actual y-values
#this shows us the data that gave us our accuracy of approx 97.68%
svm_prediction = svr_rbf.predict(x_test)
print(svm_prediction)

#print the actual y values
print(y_test)

#create a plot to visualize the predicted and the actual values
plt.figure(figsize = (12,4))
plt.plot(svm_prediction, label = 'Prediction', lw = 2, alpha = 0.7)
plt.plot(y_test, label = 'Actual', lw = 2, alpha = 0.7)
plt.title('Predicted Values vs Actual Values')
plt.ylabel('Price in USD')
plt.xlabel('Time')
plt.legend()
plt.xticks(rotation = 45)
plt.show()